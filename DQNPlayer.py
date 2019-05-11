from QPlayerBase import QPlayerBase
import datetime
import numpy as np
import tensorflow as tf
import tensorboardX


def huber_loss(labels, predictions, weights=1.0, delta=1.0, scope=None):
    """Adds a Huber Loss term to the training procedure.
    For each value x in `error=labels-predictions`, the following is calculated:
    ```
    0.5 * x^2                  if |x| <= d
    0.5 * d^2 + d * (|x| - d)  if |x| > d
    ```
    where d is `delta`.
    See: https://en.wikipedia.org/wiki/Huber_loss
    `weights` acts as a coefficient for the loss. If a scalar is provided, then
    the loss is simply scaled by the given value. If `weights` is a tensor of size
    `[batch_size]`, then the total loss for each sample of the batch is rescaled
    by the corresponding element in the `weights` vector. If the shape of
    `weights` matches the shape of `predictions`, then the loss of each
    measurable element of `predictions` is scaled by the corresponding value of
    `weights`.
    Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    delta: `float`, the point where the huber loss function
      changes from a quadratic to linear.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
    Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.
    Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or
     `predictions` is None.
    @compatibility(eager)
    The `loss_collection` argument is ignored when executing eagerly. Consider
    holding on to the return value or collecting losses via a `tf.keras.Model`.
    @end_compatibility
    """
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")
    with tf.name_scope(scope, "huber_loss", (predictions, labels, weights)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        error = tf.subtract(predictions, labels)
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        # The following expression is the same in value as
        # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
        # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
        # This is necessary to avoid doubling the gradient, since there is already a
        # nonzero contribution to the gradient from the quadratic term.
        linear = tf.subtract(abs_error, quadratic)
        losses = tf.add(
            tf.multiply(
                tf.convert_to_tensor(0.5, dtype=quadratic.dtype),
                tf.multiply(quadratic, quadratic)),
            tf.multiply(delta, linear))
    return tf.losses.compute_weighted_loss(losses, weights, scope)


tf_type = tf.float32


class DQNPlayer(QPlayerBase):
    def __init__(self):
        super(DQNPlayer, self).__init__()
        # params
        self.loss_type = 'huber'  # 'huber' or 'mse'
        self.ox_board_size = 3  # board size
        self.update_pred_network_every_n = 5  # how often to copy QNN params to target network (<1 - every train step, >=1 - every n train steps)
        self.memory_batch_size = 128  # how many samples to learn on in one train step
        self.memory_replay_size = 5000  # gather samples from last memory_replay_size steps from memory
        self.device = 'gpu'  # 'gpu' or 'cpu'
        self.hidden_layers = [80, 40]  # [x, y, z] -> means 4 fully connected layers |x -> y -> z -> output layer|
        # internal state
        self.state_size = (self.ox_board_size**2) * 2
        self.action_size = self.ox_board_size**2
        self.frames_since_pred_network_update = 0
        self.memory = []
        self._build_model(self.hidden_layers)

    # TODO: this assumes players IDs to be 1 or 2
    # state from tuple of tuples (0-no player, 1-player1, 2-player2) to 18 ints 0 or 1
    # eg. ((1, 0, 0),
    #      (0, 2, 0),     -->    100 000 100  000 010 011
    #      (1, 2, 1),)           \\\     ///
    #                           first 9 ints - 1 if there is player1 marker here else 0
    #                           last  9 ints - 1 if there is player2 marker here else 0
    def encode_state(self, state):
        flat_state = np.array([[j for j in i] for i in state]).reshape(-1)
        encoded_state = np.concatenate((flat_state == 1, flat_state == 2)) * 1
        return encoded_state

    # action int (0-8) to tuple (0-2, 0-2)
    def decode_action(self, action):
        decoded_action = (action // self.ox_board_size, action % self.ox_board_size)
        return decoded_action

    # action from tuple (0-2, 0-2) to int (0-8)
    def encode_action(self, action):
        return action[0] * self.ox_board_size + action[1]

    # one optimizer step plus write summary to logger
    def fit(self, state, target_f, learn_rate):
        [o, s] = self.tf_train_sess.run([self.tf_train_step, self.tf_train_summary], feed_dict={'x:0': state, 'y:0': target_f, 'learn_rate:0': learn_rate})
        for k, v in s.items():
            self.tf_summary_writer.add_scalar(k, v, self.tf_summary_global_step)

    # sample batch_sime experiences from memory to create learning match and apply one optimizer step
    def replay(self, batch_size, memory):
        inds = np.random.choice(len(memory), batch_size)
        states = np.array([self.encode_state(memory[i][0]) for i in inds])
        next_states = np.array([self.encode_state(memory[i][3]) for i in inds])
        actions = np.array([self.encode_action(memory[i][1]) for i in inds])
        rewards = np.array([memory[i][2] for i in inds])
        dones = np.array([memory[i][4] != 0 for i in inds])
        ndones = np.logical_not(dones)
        next_predicts = self.predict(next_states)
        target_f_batch = self.predict(states)
        targets = (rewards + self.discount_rate * np.amax(next_predicts, axis=1))
        target_f_batch[dones, actions[dones]] = rewards[dones]
        target_f_batch[ndones, actions[ndones]] = targets[ndones]

        # call self.update_pred_network_vars() every self.update_pred_network_every_n frames
        self.frames_since_pred_network_update += 1
        if self.frames_since_pred_network_update >= self.update_pred_network_every_n:
            self.frames_since_pred_network_update = 0
            self.update_pred_network_vars()

        # one optimizer step
        self.fit(states, target_f_batch, 1e-4 * self.learning_rate * batch_size)

    # copy NN weights from QNN to target network
    def update_pred_network_vars(self):
        train_vars = self.tf_train_sess.run(self.tf_train_vars)
        self.tf_pred_sess.run(self.tf_pred_vars_update, {'var'+str(i)+':0': train_vars[i] for i in range(len(train_vars))})

    # create tf graph
    def _build_model(self, hidden_layers):
        #############################
        self.tf_summary_global_step = 0
        #############################
        self.tf_train_graph = tf.Graph()
        if self.device == 'cpu':
            with self.tf_train_graph.as_default(), tf.device('/cpu:0'):
                _, self.tf_train_init, self.tf_train_step, self.tf_train_summary = self._build_network(hidden_layers, is_train=True)
        else:
            with self.tf_train_graph.as_default():
                _, self.tf_train_init, self.tf_train_step, self.tf_train_summary = self._build_network(hidden_layers, is_train=True)
        ###################
        self.tf_pred_graph = tf.Graph()
        if self.device == 'cpu':
            with self.tf_pred_graph.as_default(), tf.device('/cpu:0'):
                self.tf_pred, self.tf_pred_init, _, _ = self._build_network(hidden_layers)
        else:
            with self.tf_pred_graph.as_default():
                self.tf_pred, self.tf_pred_init, _, _ = self._build_network(hidden_layers)
        ###############################
        with self.tf_pred_graph.as_default(), tf.device('/cpu:0'):
            self.tf_train_vars = self.tf_train_graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.tf_pred_vars = self.tf_pred_graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.tf_pred_vars_update = [tf.assign(self.tf_pred_vars[i], tf.placeholder(dtype=tf_type, shape=self.tf_train_vars[i].shape, name='var'+str(i))) for i in range(len(self.tf_train_vars))]
        ###############################
        config = tf.ConfigProto(
            # device_count={'GPU': 1, 'CPU': 1}
        )
        self.tf_train_sess = tf.Session(config=config, graph=self.tf_train_graph)
        self.tf_train_sess.run(self.tf_train_init)
        filename = str(datetime.datetime.now()).replace(' ', '_').replace(':', '.') + '_' + 'train_log'
        self.tf_summary_writer = tensorboardX.SummaryWriter('./tfb_data/{}'.format(filename))
        self.tf_summary_writer.add_graph(self.tf_train_graph)

        self.tf_pred_sess = tf.Session(config=config, graph=self.tf_pred_graph)
        self.tf_pred_sess.run(self.tf_pred_init)

    # create tf nn
    def _build_network(self, hidden_layers, is_train=False):
        last_size = hidden_layers[-1]
        weight_initializer = tf.contrib.layers.xavier_initializer()
        bias_initializer = tf.contrib.layers.xavier_initializer()
        x = tf.placeholder(dtype=tf_type, shape=[None, self.state_size], name='x')
        y = tf.placeholder(dtype=tf_type, shape=[None, self.action_size], name='y')
        learn_rate = tf.placeholder(dtype=tf_type, shape=(), name='learn_rate')

        in_out_hidden_layer = x
        for size in hidden_layers:
            in_out_hidden_layer = tf.contrib.layers.fully_connected(in_out_hidden_layer, size, tf.nn.relu)

        w3 = tf.get_variable('w3', [last_size, self.action_size], tf_type, weight_initializer)
        b3 = tf.get_variable('b3', [1, self.action_size], tf_type, bias_initializer)
        l3 = tf.matmul(in_out_hidden_layer, w3) + b3
        lout = l3

        train_step = None
        train_summary = None
        if is_train:
            err = huber_loss(labels=y, predictions=lout) if self.loss_type == 'huber' \
                else tf.losses.mean_squared_error(labels=y, predictions=lout)
            train_step = tf.train.AdamOptimizer(learn_rate).minimize(err)

            train_summary = {self.loss_type: err}

        init = tf.global_variables_initializer()

        return lout, init, train_step, train_summary

    # predicted value for encoded state, returns encoded action
    def predict(self, state):
        pred = self.tf_pred_sess.run(self.tf_pred, feed_dict={'x:0': state})
        return pred

    def decide_for_action(self, state):
        action = np.argmax(self.predict(self.encode_state(state).reshape(-1, self.state_size)))
        action = self.decode_action(action)
        return action

    def update_params(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

        self.tf_summary_global_step += 1
        self.tf_summary_writer.add_scalar('Rewards', reward, self.tf_summary_global_step)
        # TODO: next lines assume done values:
        #        0 - round not finished
        #        1 - p1 won
        #        2 - p2 won
        #       -1 - tie
        win_history = np.array([m[4] for m in self.memory if m[4] != 0])
        if len(win_history):
            self.tf_summary_writer.add_scalar('WonP1', (win_history == 1).mean(), self.tf_summary_global_step)
            self.tf_summary_writer.add_scalar('Tie', (win_history == -1).mean(), self.tf_summary_global_step)
            self.tf_summary_writer.add_scalar('WonP2', (win_history == 2).mean(), self.tf_summary_global_step)

        if len(self.memory) > self.memory_batch_size*2:
            self.replay(self.memory_batch_size, self.memory[:self.memory_replay_size])
        elif len(self.memory) > 8:
            self.replay(8, self.memory)

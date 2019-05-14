import Game
import numpy as np
import random


class QPlayerBase:
    def __init__(self, num_epochs):
        # Hyparameters
        self.num_episodes = num_epochs
        self.learning_rate = 0.1
        self.discount_rate = 0.99

        self.exploration_rate = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.0005
        # logging
        self.rewards_all_episodes = None
        self.winner_all_episodes = None

    # make random action for state
    def select_random(self, state):
        action = random.choice(Game.Game.game_possibilities(state))
        return action

    # override this, should return best action based on state
    def decide_for_action(self, state):
        raise NotImplementedError()

    # override this, should update learn logic and params
    def update_params(self, state, action, reward, new_state, done):
        raise NotImplementedError()

    # start many games against random player with learning
    def learn(self):
        self.play(True)

    # start many games against random player, do not learn (do not call update_params)
    def play(self, train=False):
        self.rewards_all_episodes = []
        self.winner_all_episodes = []
        for episode in range(self.num_episodes):
            game = Game.Game()
            player_id = 1
            state = game.board
            state = tuple(tuple(x) for x in state)
            rewards_current_episode = 0

            for step in range(5):
                exploration_rate_threshold = random.uniform(0, 1)
                if exploration_rate_threshold > self.exploration_rate:
                    action = self.decide_for_action(state)
                else:
                    action = self.select_random(state)

                new_state, reward, done = game.play_game_for_training(action, player_id)
                new_state = tuple(tuple(x) for x in new_state)

                if train:
                    self.update_params(state, action, reward, new_state, done)

                state = new_state
                rewards_current_episode += reward

                if done != 0:
                    self.winner_all_episodes.append(done)
                    break

            self.exploration_rate = self.min_exploration_rate + \
                               (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)
            self.rewards_all_episodes.append(rewards_current_episode)

            # TODO: next lines assume done values:
            #        0 - round not finished
            #        1 - p1 won
            #        2 - p2 won
            #       -1 - tie
            if episode % 100 == 0:
                r = np.array(self.winner_all_episodes)
                print('{} Explore {:.3f}: Won:{:.3f} Tie: {:.3f} Lost:{:.3f}'.format(episode, self.exploration_rate, (r == 1).mean(), (r == -1).mean(), (r == 2).mean()))

        rewards_per_thousand_episodes = np.split(np.array(self.rewards_all_episodes), self.num_episodes / 1000)
        count = 1000

        print("********Average reward per thousand episodes********\n")
        for r in rewards_per_thousand_episodes:
            print('{} : Avg sum of reward:{:.3f}'.format(count, r.mean()))
            count += 1000

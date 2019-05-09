import Game
import random
import numpy as np
class QPlayer():

    def __init__(self):
        self.q_table = {}
        # Hyparameters
        self.num_episodes = 10000
        self.learning_rate = 0.1
        self.discount_rate = 0.99

        self.exploration_rate = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.001

    def select_random(self,game):
        selection = game.possibilities()
        current_loc = random.choice(selection)
        return (current_loc)

    def learn(self):

        self.rewards_all_episodes = []
        for episode in range(self.num_episodes):
            game = Game.Game()
            player_id = 1
            state = game.board
            state = tuple(tuple(x) for x in state)
            rewards_current_episode = 0

            for step in range(5):
                # Exploration-exploitation trade-off
                if(state in self.q_table):
                    exploration_rate_threshold = random.uniform(0, 1)
                    if exploration_rate_threshold > self.exploration_rate:
                        action = np.argmax(self.q_table[state])
                        action = (action//3,action%3)
                    else:
                        action = self.select_random(game)
                else:
                    self.q_table[state] = np.zeros([3,3])
                    action = self.select_random(game)


                new_state, reward, done = game.play_game_for_training(action,player_id)

                new_state = tuple(tuple(x) for x in new_state)
                state_plays = self.q_table[state]

                if(new_state not in self.q_table):
                    self.q_table[new_state] = np.zeros([3, 3])

                state_plays[action[0],action[1]] = state_plays[action[0],action[1]] * (1 - self.learning_rate) + \
                                                   self.learning_rate * (
                                                   reward + self.discount_rate * np.max(self.q_table[new_state]))



                state = new_state
                rewards_current_episode += reward

                if done != 0:
                    break

            self.exploration_rate = self.min_exploration_rate + \
                               (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)
            self.rewards_all_episodes.append(rewards_current_episode)
        rewards_per_thosand_episodes = np.split(np.array(self.rewards_all_episodes), self.num_episodes / 1000)
        count = 1000

        print("********Average reward per thousand episodes********\n")
        for r in rewards_per_thosand_episodes:
            print(count, ": ", str(sum(r / 1000)))
            count += 1000

qpl = QPlayer()
qpl.learn()
print(qpl.rewards_all_episodes)
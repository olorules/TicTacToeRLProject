from QPlayerBase import QPlayerBase
from Game import Game
import numpy as np


class QPlayer(QPlayerBase):
    def __init__(self, num_epochs=10000):
        super(QPlayer, self).__init__(num_epochs)
        self.q_table = {}

    def decide_for_action(self, state):
        # Exploration-exploitation trade-off
        if state in self.q_table:
            action = np.argmax(self.q_table[state])
            action = (action // 3, action % 3)
        else:
            self.q_table[state] = np.zeros([3, 3])
            action = self.select_random(state)
        return action

    def update_params(self, state, action, reward, new_state, done):
        if state not in self.q_table:
            self.q_table[state] = np.zeros([3, 3])

        state_plays = self.q_table[state]

        if new_state not in self.q_table:
            self.q_table[new_state] = np.zeros([3, 3])

        state_plays[action[0], action[1]] = state_plays[action[0], action[1]] * (1 - self.learning_rate) + \
                                            self.learning_rate * (
                                                    reward + self.discount_rate * np.max(self.q_table[new_state]))


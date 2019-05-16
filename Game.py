import numpy as np
import random
from time import sleep

class Game:

    def __init__(self):
        self.board = self.create_board()
        self.winner, self.counter = 0, 1
        #print(self.board)
        self.player_markers = [2, 1]

    def create_board(self):
        return (np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]))


    # Check for empty places on state
    @staticmethod
    def game_possibilities(state):
        l = []

        for i in range(len(state)):
            for j in range(len(state)):

                if state[i][j] == 0:
                    l.append((i, j))
        return (l)

    # Check for empty places on this board
    def possibilities(self):
        return Game.game_possibilities(self.board)

    # Select a random place for the player
    def random_place(self, player):
        selection = self.possibilities()
        current_loc = random.choice(selection)
        self.board[current_loc] = player
        return (self.board)

    def choosen_place(self, player,x,y):
        self.board[2-y,x] = player
        return (self.board)


    # Checks whether the player has three
    # of their marks in a horizontal row
    def row_win(self, player):
        for x in range(len(self.board)):
            win = True

            for y in range(len(self.board)):
                if self.board[x, y] != player:
                    win = False
                    continue

            if win == True:
                return (win)
        return (win)


    # Checks whether the player has three
    # of their marks in a vertical row
    def col_win(self, player):
        for x in range(len(self.board)):
            win = True

            for y in range(len(self.board)):
                if self.board[y][x] != player:
                    win = False
                    continue

            if win == True:
                return (win)
        return (win)


    # Checks whether the player has three
    # of their marks in a diagonal row
    def diag_win(self, player):
        win = True

        for x in range(len(self.board)):
            if self.board[x, x] != player:
                win = False
        return (win)


    # Evaluates whether there is
    # a winner or a tie
    def evaluate(self):
        self.winner = 0

        for player in [2, 1]:
            if (self.row_win(player) or
                    self.col_win(player) or
                    self.diag_win(player)):
                self.winner = player

        if np.all(self.board != 0) and self.winner == 0:
            self.winner = -1
        return self.winner


    def play_game_for_comparison(self,players):
        while self.winner == 0:
            #sleep(1)
            for player in zip(players,self.player_markers):
                action = player[0].decide_for_action(tuple(tuple(x) for x in self.board))
                possibilities = self.possibilities()
                possible = action in possibilities
                if not possible:
                    action = possibilities[0]
                self.board[action[0], action[1]] = player[1]
                print("Board after " + str(self.counter) + " move")
                print(self.board)
                #sleep(1)
                self.counter += 1
                self.winner = self.evaluate()
                if self.winner != 0:
                    break
        return (self.winner)
    
    
    # Main function to start the game
    def play_game(self):
        while self.winner == 0:
            #sleep(1)
            for player in self.player_markers:
                self.board = self.random_place(player)
                print("Board after " + str(self.counter) + " move")
                print(self.board)
                #sleep(1)
                self.counter += 1
                self.winner = self.evaluate()
                if self.winner != 0:
                    break
        return (self.winner)

    def play_game_for_training(self, action, player):
        possibilities = self.possibilities()
        possible = action in possibilities
        # todo: assume -0.9 reward for incorrect action
        if not possible:
            action = possibilities[0]

        self.board[action[0], action[1]] = player
        if self.evaluate() != 0:
            return self.board, 1 if possible else -0.9, self.evaluate()
        self.board = self.random_place(player+1)
        if self.evaluate() != 0:
            return self.board, -1 if possible else -0.9, self.evaluate()

        return self.board, self.evaluate_for_reward() if possible else -0.9, self.evaluate()

    # ranged for -1 to 1 exclusive, more than 0 is good
    def evaluate_for_reward(self, state=None):
        if state is None:
            state = self.board
        # TODO: maybe return heuristic score not 0, should help in learning, but its not necessary
        return 0

    @staticmethod
    def play_multigame(num_episodes, players, player_markers, trains=(False, False), alternate=False):
        rewards_all_episodes = []
        winner_all_episodes = []
        ids = [(0, 1), (1, 0)]
        for episode in range(num_episodes):
            game = Game()
            state = game.board
            state = tuple(tuple(x) for x in state)
            rewards_current_episode = 0

            done = 0
            num_moves = 0
            while done == 0:
                for id, other_id in ids:
                    player_ai, player_marker, train = players[id], player_markers[id], trains[id]
                    other_ai, other_marker, other_train = players[other_id], player_markers[other_id], trains[other_id]

                    if train:
                        action = player_ai.decide_for_action_explore(state)
                    else:
                        action = player_ai.decide_for_action(state)

                    new_state, reward, done = game.play_move_for_training(action, player_marker)
                    new_state = tuple(tuple(x) for x in new_state)
                    num_moves += 1

                    ##TODO: remove block for bad action and add tie after 10 moves
                    if train:
                        player_ai.update_params(state, action, reward, new_state, done)
                    if other_train:
                        other_ai.update_params(state, action, -reward, new_state, done)

                    state = new_state
                    rewards_current_episode += reward

                    if done != 0:
                        winner_all_episodes.append(done)
                        break

            rewards_all_episodes.append(rewards_current_episode)

            # TODO: next lines assume done values:
            #        0 - round not finished
            #        1 - p1 won
            #        2 - p2 won
            #       -1 - tie
            if episode % 100 == 0:
                r = np.array(winner_all_episodes)
                print('{} Explore {:.3f}; {:.3f}: Won:{:.3f} Tie: {:.3f} Lost:{:.3f}'.format(episode, players[0].exploration_rate, players[1].exploration_rate, (r == 1).mean(), (r == -1).mean(), (r == 2).mean()))

            if alternate:
                ids = ids[::-1]

        rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
        count = 1000

        print("********Average reward per thousand episodes********\n")
        for r in rewards_per_thousand_episodes:
            print('{} : Avg sum of reward:{:.3f}'.format(count, r.mean()))
            count += 1000

    def play_move_for_training(self, action, player_marker):
        possibilities = self.possibilities()
        possible = action in possibilities
        # todo: assume -0.9 reward for incorrect action
        if not possible:
            action = possibilities[0]
            return self.board, -1.5, 1 if player_marker == 2 else 2

        self.board[action[0], action[1]] = player_marker
        if self.evaluate() != 0:
            return self.board, 1 if possible else -1.5, self.evaluate()

        return self.board, self.evaluate_for_reward() if possible else -1.5, self.evaluate()

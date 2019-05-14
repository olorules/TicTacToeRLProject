import numpy as np
import random
from time import sleep

class Game:

    def __init__(self):
        self.board = self.create_board()
        self.winner, self.counter = 0, 1
        #print(self.board)
        self.players = [2, 1]

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
            sleep(1)
            for player in zip(players,self.players):
                #self.board = self.random_place(player)
                action = player[0].decide_for_action(tuple(tuple(x) for x in self.board))
                self.board[action[0], action[1]] = player[1]
                print("Board after " + str(self.counter) + " move")
                print(self.board)
                sleep(1)
                self.counter += 1
                self.winner = self.evaluate()
                if self.winner != 0:
                    break
        return (self.winner)
    
    
    # Main function to start the game
    def play_game(self):
        while self.winner == 0:
            sleep(1)
            for player in self.players:
                self.board = self.random_place(player)
                print("Board after " + str(self.counter) + " move")
                print(self.board)
                sleep(1)
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
            return self.board, -0.9, self.evaluate()

        self.board[action[0], action[1]] = player
        if self.evaluate() != 0:
            return self.board, 1, self.evaluate()
        self.board = self.random_place(player+1)
        if self.evaluate() != 0:
            return self.board, -1, self.evaluate()

        return self.board, self.evaluate_for_reward(), self.evaluate()

    # ranged for -1 to 1 exclusive, more than 0 is good
    def evaluate_for_reward(self, state=None):
        if state is None:
            state = self.board
        # TODO: maybe return heuristic score not 0, should help in learning, but its not necessary
        return 0


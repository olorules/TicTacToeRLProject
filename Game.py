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


    # Check for empty places on board
    def possibilities(self):
        l = []

        for i in range(len(self.board)):
            for j in range(len(self.board)):

                if self.board[i][j] == 0:
                    l.append((i, j))
        return (l)


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
            if (self.row_win( player) or
                    self.col_win(player) or
                    self.diag_win(player)):
                self.winner = player

        if np.all(self.board != 0) and self.winner == 0:
            self.winner = -1
        return self.winner


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

        self.board[action[0], action[1]] = player
        if self.evaluate() != 0:
            return self.board, 1,self.evaluate()
        self.board = self.random_place(player+1)
        if self.evaluate() != 0:
            return self.board, -1,self.evaluate()

        return self.board, 0,self.evaluate()

# Driver Code
#game = Game()
#print("Winner is: " + str(game.play_game()))
#game.q_player(1)
#game.play_game()
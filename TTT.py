import numpy as np

from GAME import GAME
import utils

class Board:
    def __init__(self, size=3):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn_num = 0
        self.current_player = -1
        self.winner = None

    def __valid_square(self, square):
        return square[0] >= 0 and square[0] < self.size and square[1] >=0 and square[1] < self.size

    def __valid_player(self, player):
        return player == 1 or player == -1

    def mark(self, square, player):
        # if self.winner is not None:
        #     # print("Game is already over, won by player {}".format(self.winner))
        #     self.winner = -player
        #     return
        # if not self.__valid_square(square):
        #     # print("{} is out of bounds for this board (size={}).".format(square, self.size))
        #     self.winner = -player
        #     return
        # if not self.__valid_player(player):
        #     # print("{} is not a valid player (must be 1 or -1)".format(player))
        #     self.winner = -player
        #     return
        if self.board[square]:
            # print("{} has already been played on this board.".format(square, self.board))
            self.winner = -player
            return

        self.board[square] = player
        self.current_player = -self.current_player
        self.turn_num += 1

        self.winner = self.__check_winner()

    def __check_winner(self):
        # Check draw
        if self.turn_num == 9:
            return 0

        # Check rows
        for row in range(self.size):
            first = self.board[row][0]
            if first and np.all(self.board[row] == first):
                return first

        # Check columns
        for col in range(self.size):
            first = self.board[:, col][0]
            if first and np.all(self.board[:, col] == first):
                return first

        # Check diagonals
        first = self.board[0][0]
        if first and np.all(self.board[np.arange(self.size), np.arange(self.size)] == first):
            return first
        first = self.board[0][-1]
        if first and np.all(self.board[np.arange(self.size), self.size - 1 - np.arange(self.size)] == first):
            return first

        return None

class TTT(GAME):
    def __init__(self, size=3):
        self.size = size
        self.board = Board(size)

    def play(self, strategy1, strategy2=None):
        if strategy2 is None:
            strategy2 = strategy1

        while not self.ended():
            # print(self.board.board.T)
            player = self.board.current_player
            if player == -1:
                action = strategy1(self.get_state())
            else:
                action = strategy2(self.get_state())
            self.move(action)

        # print(self.board.board)
        # if self.board.winner != 0:
        #     print("Winner: {}".format(self.get_winner()))
        # else:
        #     print("Draw.")

        return self.get_winner()

    def get_state(self):
        return self.board.board.ravel()

    def move(self, action):
        square = (action // self.size, action % self.size)
        player = self.board.current_player
        self.board.mark(square, player)
        # print(self.board.board.T)

    def get_winner(self):
        return self.board.winner

    def ended(self):
        return self.board.winner is not None

def TTT_input(state):
    result = None
    while not result:
        try:
            square_str = input("Player {}, pick a square: ".format(board.current_player))
            square_str_list = square_str.split(",")
            if len(square_str_list) != 2:
                raise ValueError("Could not parse (x, y) coordinates from input.")
            result = [utils.input_to_int(entry) for entry in square_str_list]
        except ValueError as err:
            print(err)
            continue
    return 3 * result[0] + result[1]

def TTT_random(state):
    valid_actions = np.where(state == 0)[0]
    return np.random.choice(valid_actions)

if __name__ == "__main__":
    results = []
    for _ in range(10000):
        test = TTT()
        results.append(test.play(strategy1=TTT_random))
    print(sum(results))

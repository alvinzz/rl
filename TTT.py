import numpy as np

from GAME import GAME
import utils

class Board:
    def __init__(self, size=3):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.move_num = 0
        self.current_player = -1
        self.winner = None

    def __valid_square(self, square):
        return square[0] >= 0 and square[0] < self.size and square[1] >=0 and square[1] < self.size

    def __valid_player(self, player):
        return player == 1 or player == -1

    def mark(self, square, player):
        if self.winner:
            print("Game is already over, won by player {}".format(self.winner))
            return False
        if not self.__valid_square(square):
            print("{} is out of bounds for this board (size={}).".format(square, self.size))
            return False
        if not self.__valid_player(player):
            print("{} is not a valid player (must be 1 or -1)".format(player))
            return False
        if self.board[square]:
            print("{} has already been played on this board.".format(square, self.board))
            return False

        self.board[square] = player
        self.current_player = -self.current_player
        self.move_num += 1

        winner = self.__check_winner()
        if winner is not None:
            self.winner = winner

        return True

    def __check_winner(self):
        # Check draw
        if self.move_num == 9:
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
        self.board = Board(self.size)
        while self.board.winner is None:
            player = self.board.current_player
            if player == -1:
                square = strategy1(self.board)
            else:
                square = strategy2(self.board)
            if not self.board.mark(square, player):
                # print("Illegal move. Winner is {}.".format(-player))
                return -player

        # print(self.board.board)
        # if self.board.winner != 0:
        #     print("Winner: {}".format(self.board.winner))
        # else:
        #     print("Draw.")

        return self.board.winner

def TTT_input(board):
    result = None
    while not result:
        print(board.board.T)
        try:
            square_str = input("Player {}, pick a square: ".format(board.current_player))
            square_str_list = square_str.split(",")
            if len(square_str_list) != 2:
                raise ValueError("Could not parse (x, y) coordinates from input.")
            result = [utils.input_to_int(entry) for entry in square_str_list]
        except ValueError as err:
            print(err)
            continue
    return tuple(result)

def TTT_random(board):
    valid_squares = np.vstack(np.where(board.board == 0)).T
    index = np.random.randint(len(valid_squares))
    return tuple(valid_squares[index])

if __name__ == "__main__":
    test = TTT(3)
    results = []
    for _ in range(10000):
        results.append(test.play(strategy1=TTT_random))
    print(sum(results))

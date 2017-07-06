import numpy as np

from GAME import GAME
import utils

class Board:
    def __init__(self, size=3):
        assert type(size) is int, "Size must be an integer."
        self.size = size
        self.state = np.zeros(2 * self.size * self.size, dtype=np.uint8)
        self.turn_num = 0
        self.current_player = 0
        self.winner = None

    def mark(self, square):
        index = self.size * square[1] + square[0]

        if self.state[index] or self.state[index + self.size * self.size]:
            self.winner = 1 - self.current_player
            return

        if self.current_player == 0:
            self.state[index] = 1
        else:
            self.state[index + self.size * self.size] = 1

        self.turn_num += 1
        self.current_player = 1 - self.current_player

        self.winner = self.__check_winner()

    def __check_winner(self):
        # Check rows and cols
        for i in range(self.size):
            row = self.state[np.arange(self.size) + self.size * i]
            col = self.state[np.arange(self.size) * self.size + i]
            if np.all(row) or np.all(col):
                return 0

        for i in range(self.size):
            row = self.state[np.arange(self.size) + self.size * i + self.size * self.size]
            col = self.state[np.arange(self.size) * self.size + i + self.size * self.size]
            if np.all(row) or np.all(col):
                return 1

        # Check diagonals
        if np.all(self.state[np.arange(self.size) * (self.size + 1)]):
            return 0
        if np.all(self.state[np.arange(self.size) * (self.size - 1) + self.size - 1]):
            return 0
        if np.all(self.state[np.arange(self.size) * (self.size + 1) + self.size * self.size]):
            return 1
        if np.all(self.state[np.arange(self.size) * (self.size - 1) + self.size - 1 + self.size * self.size]):
            return 1

        # Check draw: nobody won (from above), and the 9th move has been made
        if self.turn_num == 9:
            return 0.5

        # Game is still in progress
        return None

    def display(self):
        player1_rows = [i // self.size for i in np.where(self.state[:self.size * self.size])][0]
        player1_cols = [i % self.size for i in np.where(self.state[:self.size * self.size])][0]
        player2_rows = [i // self.size for i in np.where(self.state[self.size * self.size:])][0]
        player2_cols = [i % self.size for i in np.where(self.state[self.size * self.size:])][0]
        res = np.full((self.size, self.size), '_', dtype=np.object_)
        for row, col in zip(player1_rows, player1_cols):
            res[row, col] = 'X'
        for row, col in zip(player2_rows, player2_cols):
            res[row, col] = 'O'
        print(res)

def TTT_input(state):
    result = None
    while not result:
        try:
            square_str = input("Pick a square: ")
            square_str_list = square_str.split(",")
            if len(square_str_list) != 2:
                raise ValueError("Could not parse (x, y) coordinates from input.")
            result = [utils.input_to_int(entry) for entry in square_str_list]
        except ValueError as err:
            print(err)
            continue
    return 3 * result[1] + result[0]

def TTT_random(state):
    num_actions = int(len(state) / 2)
    valid_actions = np.arange(num_actions)
    valid_actions = list(filter(lambda square: state[square] == 0 and state[square + num_actions] == 0, valid_actions))
    return np.random.choice(valid_actions)

class TTT(GAME):
    def __init__(self, size=3, display=False):
        self.size = size
        self.board = Board(size)
        self.display = display

    def play(self, strategy1=TTT_random, strategy2=TTT_input):
        if strategy2 is None:
            strategy2 = strategy1

        while not self.ended():
            if self.display:
                self.board.display()
            player = self.board.current_player
            if player == 0:
                action = strategy1(self.board.state)
            else:
                action = strategy2(np.hstack((self.board.state[self.size * self.size:], self.board.state[:self.size * self.size])))
            self.move(action)

        if self.display:
            self.board.display()
        return self.get_winner()

    def move(self, action):
        square = (action % self.size, action // self.size)
        self.board.mark(square)

    def get_winner(self):
        return self.board.winner

    def get_state(self):
        return self.board.state

    def ended(self):
        return self.board.winner is not None

if __name__ == "__main__":
    import pickle
    model = pickle.load(open("./TTT_batch_100_110000.p", "rb"))

    def create_nn_strategy(model):
        def nn_strategy(state):
            return utils.run_model(model, state, stochastic=False)
        return nn_strategy

    nn_strategy = create_nn_strategy(model)

    results = []
    for _ in range(5000):
        test = TTT(3, display=False)
        result = 1 - test.play(strategy1=nn_strategy, strategy2=TTT_random)
        results.append(result)
        test = TTT(3, display=False)
        result = test.play(strategy1=TTT_random, strategy2=nn_strategy)
        results.append(result)
    print(sum(results))

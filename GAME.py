import abc

class GAME(metaclass=abc.ABCMeta):
    # plays the game. takes a strategy as argument, returns the final score
    @abc.abstractmethod
    def play(self):
        pass

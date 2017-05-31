import abc

class GAME(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def move(self, action):
        pass

    @abc.abstractmethod
    def ended(self):
        pass

    @abc.abstractmethod
    def get_state(self):
        pass

    @abc.abstractmethod
    def get_winner(self):
        pass

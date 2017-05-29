import tensorflow as tf
import numpy as np

from TTT import TTT

class TTT_RL:
    def __init__(self):
        self.game = TTT(3)

        self.hidden_dims = [50]
        self.batch_size = 100
        self.epochs = 1000
        self.learning_rate = 1e-4
        # self.discount_factor = 0.9 # gamma
        # self.exploration_param = 0.5 # epsilon
        # self.exploration_decay = self.exploration_param / self.epochs
        # self.max_grad = 1.0
        self.l2_weight = 0.001
        # self.dropout_prob = 0.1
        # self.past_strategies = []

        self.session = tf.Session()
        self.trainable = {}
        self.outputs = {}

        # input
        self.input = tf.placeholder(tf.float32, shape=[None, 9])

        # hidden layers
        self.outputs["h-1"] = self.input
        for i in np.arange(len(self.hidden_dims)):
            self.trainable["W{}".format(i)] = tf.get_variable("W{}".format(i), [9, self.hidden_dims[i]], initializer=tf.contrib.layers.xavier_initializer()).initialized_value()
            self.trainable["b{}".format(i)] = tf.get_variable("b{}".format(i), [self.hidden_dims[i]], initializer=tf.zeros_initializer()).initialized_value()
            self.outputs["h{}".format(i)] = tf.nn.relu(tf.matmul(self.outputs["h{}".format(i - 1)], self.trainable["W{}".format(i)]) + self.trainable["b{}".format(i)])

        # ouput
        i = len(self.hidden_dims)
        self.trainable["W{}".format(i)] = tf.get_variable("W{}".format(i), [self.hidden_dims[-1], 9], initializer=tf.contrib.layers.xavier_initializer()).initialized_value()
        self.trainable["b{}".format(i)] = tf.get_variable("b{}".format(i), [9], initializer=tf.zeros_initializer()).initialized_value()
        self.probs = tf.nn.softmax(tf.matmul(self.outputs["h{}".format(i - 1)], self.trainable["W{}".format(i)]) + self.trainable["b{}".format(i)])[0]

        self.l2_loss = tf.reduce_sum([tf.reduce_sum(tf.square(self.trainable["W{}".format(i)])) for i in np.arange(len(self.hidden_dims) + 1)])
        # play a random number of games as player 1 and the remainder as player -1
        self.strat_loss = 0
        player_1_games = np.random.randint(self.batch_size)
        for _ in range(player_1_games):
            self.strat_loss -= self.game.play(self.strategy)
        for _ in range(self.batch_size - player_1_games):
            self.strat_loss += self.game.play(self.strategy)
        self.strat_loss /= 100
        self.loss = self.strat_loss + self.l2_weight * self.l2_loss
        # yeah ok we fix this tomorrow
        self.optimize = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

    def strategy(self, board):
        probs = self.session.run(self.probs, feed_dict={self.input: np.expand_dims(board.board.ravel(), axis=0)})
        sampled = np.random.choice(len(probs), 1, p=probs)[0]
        return sampled // 3, sampled % 3

    def train(self):
        for _ in range(self.epochs):
            self.session.run([self.optimize, self.loss])

if __name__ == "__main__":
    test = TTT_RL()
    test.train()

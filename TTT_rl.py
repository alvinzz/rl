import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from TTT import TTT, TTT_random

class TTT_RL:
    def __init__(self):
        self.game = TTT
        self.state_space_size = 9

        self.hidden_dims = [100]
        self.batch_size = 100
        self.epochs = 10000
        self.learning_rate = 1e-4
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.discount_factor = 0.9 # gamma
        self.exploration_param = 0.4 # epsilon
        self.exploration_decay = self.exploration_param / self.epochs
        self.l2_weight = 0.001
        # self.dropout_prob = 0.0
        # self.past_strategies = []

        self.session = tf.Session()
        self.trainable = {}
        self.outputs = {}

        # input
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.state_space_size], name="inputs")

        # hidden layers
        self.trainable["W0"] = tf.get_variable("W0", [self.state_space_size, self.hidden_dims[0]], initializer=tf.contrib.layers.xavier_initializer()).initialized_value()
        self.trainable["b0"] = tf.get_variable("b0", [self.hidden_dims[0]], initializer=tf.zeros_initializer()).initialized_value()
        self.outputs["h0"] = tf.nn.relu(tf.matmul(self.inputs, self.trainable["W0"]) + self.trainable["b0"])

        for i in np.arange(1, len(self.hidden_dims)):
            self.trainable["W{}".format(i)] = tf.get_variable("W{}".format(i), [self.hidden_dims[i - 1], self.hidden_dims[i]], initializer=tf.contrib.layers.xavier_initializer()).initialized_value()
            self.trainable["b{}".format(i)] = tf.get_variable("b{}".format(i), [self.hidden_dims[i]], initializer=tf.zeros_initializer()).initialized_value()
            self.outputs["h{}".format(i)] = tf.nn.relu(tf.matmul(self.outputs["h{}".format(i - 1)], self.trainable["W{}".format(i)]) + self.trainable["b{}".format(i)])

        # ouput
        i = len(self.hidden_dims)
        self.trainable["W{}".format(i)] = tf.get_variable("W{}".format(i), [self.hidden_dims[-1], self.state_space_size], initializer=tf.contrib.layers.xavier_initializer()).initialized_value()
        self.trainable["b{}".format(i)] = tf.get_variable("b{}".format(i), [self.state_space_size], initializer=tf.zeros_initializer()).initialized_value()
        self.logprobs = tf.matmul(self.outputs["h{}".format(i - 1)], self.trainable["W{}".format(i)]) + self.trainable["b{}".format(i)]
        self.probs = tf.nn.softmax(self.logprobs)

        # store rollout results
        self.rollout_actions = tf.placeholder(tf.int32, shape=[None], name="rollout_actions")
        self.discounted_rewards = tf.placeholder(tf.float32, shape=[None], name="discounted_rewards")

        # get losses, apply policy gradient
        self.gradients = []
        self.l2_loss = tf.reduce_sum([tf.reduce_sum(tf.square(self.trainable["W{}".format(i)])) for i in np.arange(len(self.hidden_dims) + 1)])
        for index in range(self.batch_size):
            self.policy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logprobs[index], labels=self.rollout_actions[index]))
            self.loss = self.policy_loss + self.l2_weight * self.l2_loss
            self.gradients.append(self.optimizer.compute_gradients(self.loss))

        self.policy_gradient = []
        for i, gradient in enumerate(self.gradients):
            for j, (grad, var) in enumerate(gradient):
                if i == 0:
                    self.policy_gradient.append([grad * self.discounted_rewards[i], var])
                else:
                    self.policy_gradient[j][0] += grad * self.discounted_rewards[i]

        self.train_op = self.optimizer.apply_gradients(self.policy_gradient)

    def sample_action(self, state):
        if np.random.rand() <= self.exploration_param:
            sampled = np.random.randint(self.state_space_size)
        else:
            probs = self.session.run(self.probs, feed_dict={self.inputs: np.expand_dims(state, axis=0)})[0]
            sampled = np.random.choice(self.state_space_size, 1, p=probs)[0]
        return sampled

    def strategy(self, state):
        probs = self.session.run(self.probs, feed_dict={self.inputs: np.expand_dims(state, axis = 0)})[0]
        return np.argmax(probs)

    def train(self):
        self.session.run(tf.global_variables_initializer())

        for epoch in range(self.epochs):
            if epoch % 1 == 0:
                print("Epoch {}, win % against random={}%".format(epoch, round(self.test() / 10, 1)))
            inputs = []
            rollout_actions = []
            discounted_rewards = []

            for batch_num in range(self.batch_size):
                batch_rewards = []

                # change this to sample from past_strategies
                other_strategy = np.random.choice([self.strategy, TTT_random])

                # play a game a record the states, actions, and rewards
                game = self.game()
                turn_num = 0
                while not game.ended():
                    state = game.get_state()
                    # play half of the games going first and half going second
                    if turn_num % 2 == batch_num % 2:
                        if turn_num > 1:
                            # nobody won in last pair of moves
                            batch_rewards.append(0)

                        inputs.append(state)

                        action = self.sample_action(state)
                        rollout_actions.append(action)
                        game.move(action)
                    else:
                        other_strategy_action = other_strategy(state)
                        game.move(other_strategy_action)

                    turn_num += 1

                # we went first and game is over
                if batch_num % 2 == 0:
                    batch_rewards.append(-game.get_winner())
                # we went second and game is over
                else:
                    batch_rewards.append(game.get_winner())

                # compute discounted rewards
                batch_discounted_rewards = []
                cum_reward = 0
                for reward in reversed(batch_rewards):
                    cum_reward = reward + self.discount_factor * cum_reward
                    batch_discounted_rewards.append(cum_reward)

                # normalize the discounted rewards
                batch_discounted_rewards -= np.mean(batch_discounted_rewards)
                if np.std(batch_discounted_rewards):
                    batch_discounted_rewards /= np.std(batch_discounted_rewards)

                discounted_rewards.extend(batch_discounted_rewards)

            _ = self.session.run(self.train_op, feed_dict={
                self.inputs: np.array(inputs, dtype=np.float32),
                self.rollout_actions: np.array(rollout_actions, dtype=np.int32),
                self.discounted_rewards: np.array(discounted_rewards, dtype=np.float32)
            })

            self.exploration_param = max(self.exploration_param - self.exploration_decay, 0)

        print("Finished training, win % against random is {}%".format(round(self.test() / 10, 1)))

    def test(self):
        results = []
        for _ in range(500):
            test = self.game()
            results.append(-test.play(strategy1=self.strategy, strategy2=TTT_random))
        for _ in range(500):
            test = self.game()
            results.append(test.play(strategy2=self.strategy, strategy1=TTT_random))
        # number of games won by our strategy
        return sum(results) / 2 + 500

if __name__ == "__main__":
    test = TTT_RL()
    test.train()

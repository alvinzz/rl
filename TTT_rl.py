import tensorflow as tf
import numpy as np
import pickle
import utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from TTT import TTT, TTT_random

# notes: with exploration=0, hidden_dims=[50], batch=100, convergence at around 50000 epochs w/ 90% winrate vs random
# TODO: get play against self to work with comparable performance!
# TODO: get deeper/larger networks to work
# TODO: TRPO

class TTT_RL:
    def __init__(self):
        self.game = TTT
        self.state_dim = 18
        self.num_actions = 9

        self.hidden_dims = [50, 50]
        self.batch_size = 100
        self.epochs = 1000000
        self.learning_rate = 1e-4
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.discount_factor = 0.9 # gamma
        # we do sampling already, so exploration not necessary?
        self.exploration_param = 0 # 0.5 # epsilon
        self.exploration_decay = 0 # self.exploration_param / (self.epochs * 0.8)
        self.l2_weight = 0.001
        # self.dropout_prob = 0.0 # not necessary, seeing as overfitting is not an issue for TTT
        self.past_models = [None]
        self.past_strategies = [self.generate_ith_nn_strat(0)]

        self.session = tf.Session()
        self.trainable = {}
        self.outputs = {}

        # input
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.state_dim], name="inputs")

        # hidden layers
        self.trainable["W0"] = tf.get_variable("W0", [self.state_dim, self.hidden_dims[0]], initializer=tf.contrib.layers.xavier_initializer())
        self.trainable["b0"] = tf.get_variable("b0", [self.hidden_dims[0]], initializer=tf.zeros_initializer())
        self.outputs["h0"] = tf.nn.relu(tf.matmul(self.inputs, self.trainable["W0"]) + self.trainable["b0"])

        for i in np.arange(1, len(self.hidden_dims)):
            self.trainable["W{}".format(i)] = tf.get_variable("W{}".format(i), [self.hidden_dims[i - 1], self.hidden_dims[i]], initializer=tf.contrib.layers.xavier_initializer())
            self.trainable["b{}".format(i)] = tf.get_variable("b{}".format(i), [self.hidden_dims[i]], initializer=tf.zeros_initializer())
            self.outputs["h{}".format(i)] = tf.nn.relu(tf.matmul(self.outputs["h{}".format(i - 1)], self.trainable["W{}".format(i)]) + self.trainable["b{}".format(i)])

        # ouput
        i = len(self.hidden_dims)
        self.trainable["W{}".format(i)] = tf.get_variable("W{}".format(i), [self.hidden_dims[-1], self.num_actions], initializer=tf.contrib.layers.xavier_initializer())
        self.trainable["b{}".format(i)] = tf.get_variable("b{}".format(i), [self.num_actions], initializer=tf.zeros_initializer())
        self.logprobs = tf.matmul(self.outputs["h{}".format(i - 1)], self.trainable["W{}".format(i)]) + self.trainable["b{}".format(i)]
        self.probs = tf.nn.softmax(self.logprobs)

        # get rollout actions and rewards
        self.rollout_actions = tf.placeholder(tf.int32, shape=[None], name="rollout_actions")
        self.discounted_rewards = tf.placeholder(tf.float32, shape=[None], name="discounted_rewards")
        self.one_hot_rollout_actions = tf.one_hot(self.rollout_actions, depth=self.num_actions, dtype=tf.float32)

        # get losses, apply policy gradient
        self.l2_loss = tf.reduce_sum([tf.reduce_sum(tf.square(self.trainable["W{}".format(i)])) for i in np.arange(len(self.hidden_dims) + 1)])
        self.policy_losses = -tf.reduce_sum(self.probs * self.one_hot_rollout_actions, reduction_indices=1) * self.discounted_rewards
        self.loss = self.l2_weight * self.l2_loss + self.policy_losses
        self.train_op = self.optimizer.minimize(self.loss)

    def sample_action(self, state):
        if np.random.rand() <= self.exploration_param:
            sampled = np.random.randint(self.num_actions)
        else:
            probs = self.session.run(self.probs, feed_dict={self.inputs: np.expand_dims(self.process_state(state), axis=0)})[0]
            sampled = np.random.choice(self.num_actions, p=probs)
        return sampled

    def strategy(self, state):
        probs = self.session.run(self.probs, feed_dict={self.inputs: np.expand_dims(self.process_state(state), axis = 0)})[0]
        return np.argmax(probs)

    def train(self):
        self.session.run(tf.global_variables_initializer())

        self.save_freq = 1000
        self.eval_freq = 100
        self.add_to_past = 0 # needs to be multiple of eval_freq

        self.epoch_score = -self.eval_freq * self.batch_size
        print("Starting training.")
        for epoch in range(self.epochs):
            if self.add_to_past and epoch % self.add_to_past == 0 and self.epoch_score > 0:
                print("Adding this iteration to past models.")
                model = self.session.run(self.trainable)
                self.past_models.append(model)
                self.past_strategies.append(self.generate_ith_nn_strat(len(self.past_models) - 1))
            if epoch % self.save_freq == 0:
                print("Saving model.")
                model = self.session.run(self.trainable)
                pickle.dump(model, open('TTT_50_50_batch_100_{}.p'.format(epoch), 'wb'))
            if epoch % self.eval_freq == 0:
                print("Epoch {}, win % against past iterations={}%".format(epoch, round(100 * (self.epoch_score / 2 / self.eval_freq / self.batch_size + 0.5), 1)))
                self.epoch_score = 0

            inputs = []
            rollout_actions = []
            discounted_rewards = []

            for batch_num in range(self.batch_size):
                batch_rewards = []

                # sample from past_strategies
                other_strategy = np.random.choice(self.past_strategies)

                turn_num = 0

                # play half of the games going first and half going second
                going_first = (batch_num % 2 == 0)
                if going_first:
                    game = self.game(3, -1, 1)
                else:
                    game = self.game(3, 1, -1)

                while not game.ended():
                    state = game.get_state()

                    if turn_num % 2 == batch_num % 2:
                        if turn_num > 1:
                            # nobody won in last pair of moves
                            batch_rewards.append(0)

                        state = self.process_state(state)

                        inputs.append(state)

                        action = self.sample_action(state)
                        rollout_actions.append(action)
                        game.move(action)
                    else:
                        state = -state
                        other_strategy_action = other_strategy(state)
                        game.move(other_strategy_action)

                    turn_num += 1

                self.epoch_score -= game.get_winner()
                batch_rewards.append(-game.get_winner())

                # compute discounted rewards
                batch_discounted_rewards = []
                cum_reward = 0
                for reward in reversed(batch_rewards):
                    cum_reward = reward + self.discount_factor * cum_reward
                    batch_discounted_rewards.append(cum_reward)
                batch_discounted_rewards = list(reversed(batch_discounted_rewards))
                discounted_rewards.extend(batch_discounted_rewards)

            # normalize the discounted rewards
            discounted_rewards -= np.mean(discounted_rewards)
            if np.std(discounted_rewards):
                discounted_rewards /= np.std(discounted_rewards)

            _ = self.session.run(self.train_op, feed_dict={
                self.inputs: np.array(inputs, dtype=np.float32),
                self.rollout_actions: np.array(rollout_actions, dtype=np.int32),
                self.discounted_rewards: np.array(discounted_rewards, dtype=np.float32)
            })

            # update exploration parameter
            self.exploration_param = max(self.exploration_param - self.exploration_decay, 0)

        print("Finished training, final win % against past iterations is {}%".format(epoch, round(100 * (self.epoch_score / 2 / self.eval_freq / self.batch_size + 0.5), 1)))

    def process_state(self, state):
        ones = np.where(state == 1)[0]
        negs = np.where(state == -1)[0]
        result = np.zeros(18)
        result[ones] = 1
        result[negs + 9] = 1
        return result

    def generate_ith_nn_strat(self, i):
        def ith_nn_strat(state):
            model = self.past_models[i]
            if model is None:
                return TTT_random(state)
            else:
                return utils.run_model(model, self.process_state(state))
        return ith_nn_strat

if __name__ == "__main__":
    test = TTT_RL()
    test.train()

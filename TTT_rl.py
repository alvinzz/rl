import tensorflow as tf
import numpy as np
import pickle
import utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from TTT import TTT, TTT_random

# notes: with hidden_dims=[50], batch=100, convergence at around 50000 epochs w/ 90% winrate vs random
# TODO: get play against self to work with comparable performance!
# TODO: get deeper/larger networks to work
# TODO: TRPO

class TTT_RL:
    def __init__(self):
        self.game = TTT
        self.state_dim = 18
        self.action_dim = 9

        self.hidden_dims = [50]
        self.batch_size = 100
        self.epochs = 1000000
        self.learning_rate = 1e-4
        self.policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.discount_factor = 0.9 # gamma
        self.l2_weight = 0.001
        self.past_models = [None]
        self.past_strategies = [self.generate_ith_nn_strat(0)]

        self.session = tf.Session()
        self.policy_network = {}
        self.value_network = {}
        self.policy_activations = {}
        self.value_activations = {}

        # input
        self.state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name="inputs")

        # hidden layers
        self.policy_network["W0"] = tf.get_variable("policy_W0", [self.state_dim, self.hidden_dims[0]], initializer=tf.contrib.layers.xavier_initializer())
        self.policy_network["b0"] = tf.get_variable("policy_b0", [self.hidden_dims[0]], initializer=tf.zeros_initializer())
        self.policy_activations["h0"] = tf.nn.relu(tf.matmul(self.state, self.policy_network["W0"]) + self.policy_network["b0"])
        self.value_network["W0"] = tf.get_variable("value_W0", [self.state_dim, self.hidden_dims[0]], initializer=tf.contrib.layers.xavier_initializer())
        self.value_network["b0"] = tf.get_variable("value_b0", [self.hidden_dims[0]], initializer=tf.zeros_initializer())
        self.value_activations["h0"] = tf.nn.relu(tf.matmul(self.state, self.value_network["W0"]) + self.value_network["b0"])

        for i in np.arange(1, len(self.hidden_dims)):
            self.policy_network["W{}".format(i)] = tf.get_variable("policy_W{}".format(i), [self.hidden_dims[i - 1], self.hidden_dims[i]], initializer=tf.contrib.layers.xavier_initializer())
            self.policy_network["b{}".format(i)] = tf.get_variable("policy_b{}".format(i), [self.hidden_dims[i]], initializer=tf.zeros_initializer())
            self.policy_activations["h{}".format(i)] = tf.nn.relu(tf.matmul(self.policy_activations["h{}".format(i - 1)], self.policy_network["W{}".format(i)]) + self.policy_network["b{}".format(i)])
            self.value_network["W{}".format(i)] = tf.get_variable("value_W{}".format(i), [self.hidden_dims[i - 1], self.hidden_dims[i]], initializer=tf.contrib.layers.xavier_initializer())
            self.value_network["b{}".format(i)] = tf.get_variable("value_b{}".format(i), [self.hidden_dims[i]], initializer=tf.zeros_initializer())
            self.value_activations["h{}".format(i)] = tf.nn.relu(tf.matmul(self.value_activations["h{}".format(i - 1)], self.value_network["W{}".format(i)]) + self.value_network["b{}".format(i)])

        # ouput
        i = len(self.hidden_dims)
        self.policy_network["W{}".format(i)] = tf.get_variable("policy_W{}".format(i), [self.hidden_dims[-1], self.action_dim], initializer=tf.contrib.layers.xavier_initializer())
        self.policy_network["b{}".format(i)] = tf.get_variable("policy_b{}".format(i), [self.action_dim], initializer=tf.zeros_initializer())
        self.policy_output = tf.matmul(self.policy_activations["h{}".format(i - 1)], self.policy_network["W{}".format(i)]) + self.policy_network["b{}".format(i)]
        self.probs = tf.nn.softmax(self.policy_output)
        self.value_network["W{}".format(i)] = tf.get_variable("value_W{}".format(i), [self.hidden_dims[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
        self.value_network["b{}".format(i)] = tf.get_variable("value_b{}".format(i), [1], initializer=tf.zeros_initializer())
        self.value = tf.matmul(self.value_activations["h{}".format(i - 1)], self.value_network["W{}".format(i)]) + self.value_network["b{}".format(i)]

        # rollout actions and rewards
        self.rollout_actions = tf.placeholder(tf.int32, shape=[None], name="rollout_actions")
        self.discounted_rewards = tf.placeholder(tf.float32, shape=[None], name="discounted_rewards")
        self.one_hot_rollout_actions = tf.one_hot(self.rollout_actions, depth=self.action_dim, dtype=tf.float32)
        self.advantage = tf.subtract(self.discounted_rewards, self.value)

        # get losses, apply policy gradient
        self.policy_l2_loss = tf.reduce_sum([tf.reduce_sum(tf.square(self.policy_network["W{}".format(i)])) for i in np.arange(len(self.hidden_dims) + 1)])
        self.policy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.probs, 1e-10, 1)) * self.one_hot_rollout_actions, reduction_indices=1) * self.advantage
        self.value_l2_loss = tf.reduce_sum([tf.reduce_sum(tf.square(self.value_network["W{}".format(i)])) for i in np.arange(len(self.hidden_dims) + 1)])
        self.value_loss = tf.reduce_sum(tf.square(self.discounted_rewards - self.value))
        self.policy_train_op = self.policy_optimizer.minimize(self.l2_weight * self.policy_l2_loss + self.policy_loss)
        self.value_train_op = self.value_optimizer.minimize(self.l2_weight * self.value_l2_loss + self.value_loss)

    def sample_action(self, state):
        probs = self.session.run(self.probs, feed_dict={self.state: np.expand_dims(state, axis=0)})[0]
        sampled = np.random.choice(self.action_dim, p=probs)
        return sampled

    def get_baseline_reward(self, state):
        return self.session.run(self.value, feed_dict={self.state: np.expand_dims(state, axis=0)})[0][0]

    def train(self):
        self.session.run(tf.global_variables_initializer())

        self.save_freq = 1000
        self.eval_freq = 100
        self.add_to_past = False # needs to be multiple of eval_freq

        self.epoch_score = 0
        print("Starting training.")
        for epoch in range(self.epochs):
            if self.add_to_past and epoch % self.add_to_past == 0 and self.epoch_score > 0:
                print("Adding this iteration to past models.")
                model = self.session.run(self.policy_network)
                self.past_models.append(model)
                self.past_strategies.append(self.generate_ith_nn_strat(len(self.past_models) - 1))
            if epoch % self.save_freq == 0:
                print("Saving model.")
                model = self.session.run(self.policy_network)
                pickle.dump(model, open('TTT_test_{}.p'.format(epoch), 'wb'))
            if epoch % self.eval_freq == 0:
                print("Epoch {}, win % against past iterations={}%".format(epoch, round(100 * (self.epoch_score / (self.eval_freq * self.batch_size)), 1)))
                self.epoch_score = 0

            rollout_states = []
            rollout_actions = []
            discounted_rewards = []
            baseline_rewards = []

            for batch_num in range(self.batch_size):
                batch_rewards = []

                # sample from past_strategies
                # other_strategy = np.random.choice(self.past_strategies)
                other_strategy = TTT_random

                # play half of the games going first and half going second
                game = self.game(3)
                going_first = np.random.rand() < 0.5
                our_turn = going_first
                while True:
                    state = game.get_state()
                    if our_turn:
                        # the reward from the last move we made is 0 because there was no winner
                        if np.sum(state) > 1:
                            batch_rewards.append(0)
                        if not going_first:
                            state = np.hstack((state[9:], state[:9]))
                        rollout_states.append(state)
                        action = self.sample_action(state)
                        game.move(action)
                        rollout_actions.append(action)
                        baseline_rewards.append(self.get_baseline_reward(state))
                        if game.ended():
                            if going_first:
                                batch_rewards.append(1 - game.get_winner())
                            else:
                                batch_rewards.append(game.get_winner())
                            break
                    else:
                        if going_first:
                            state = np.hstack((state[9:], state[:9]))
                        game.move(other_strategy(state))
                        if game.ended():
                            if going_first:
                                batch_rewards.append(1 - game.get_winner())
                            else:
                                batch_rewards.append(game.get_winner())
                            break
                    our_turn = not our_turn

                self.epoch_score += batch_rewards[-1]

                # compute discounted rewards
                discounted_rewards.extend(utils.discount_rewards(batch_rewards, discount_factor=self.discount_factor))

            self.session.run([self.policy_train_op, self.value_train_op], feed_dict={
                self.state: np.array(rollout_states, dtype=np.float32),
                self.rollout_actions: np.array(rollout_actions, dtype=np.int32),
                self.discounted_rewards: np.array(discounted_rewards, dtype=np.float32),
            })

    def generate_ith_nn_strat(self, i):
        def ith_nn_strat(state):
            model = self.past_models[i]
            if model is None:
                return TTT_random(state)
            else:
                return utils.run_model(model, self.process_state(state), stochastic=True)
        return ith_nn_strat

if __name__ == "__main__":
    test = TTT_RL()
    test.train()

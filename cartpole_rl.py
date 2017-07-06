import tensorflow as tf
import numpy as np
import gym

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CARTPOLE_RL:
    def __init__(self):
        env_name = 'CartPole-v0'
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.hidden_dims = [20]
        self.batch_size = 1
        self.epochs = 10000
        self.learning_rate = 1e-4
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.discount_factor = 0.99 # gamma
        self.exploration_param = 0 #0.5 # epsilon
        self.exploration_decay = 0 #self.exploration_param / self.epochs
        self.l2_weight = 0.001
        # self.dropout_prob = 0.0
        # self.past_strategies = []

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

        # store rollout results
        self.rollout_actions = tf.placeholder(tf.int32, shape=[None], name="rollout_actions")
        self.discounted_rewards = tf.placeholder(tf.float32, shape=[None], name="discounted_rewards")

        # get losses, apply policy gradient
        self.l2_loss = tf.reduce_sum([tf.reduce_sum(tf.square(self.trainable["W{}".format(i)])) for i in np.arange(len(self.hidden_dims) + 1)])
        self.policy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logprobs, labels=self.rollout_actions))
        self.loss = self.policy_loss + self.l2_weight * self.l2_loss
        grads = self.optimizer.compute_gradients(self.loss)
        # put gradients in fixed ordering: W0, b0, W1, b1, ...
        self.grad_dict = {}
        for k in grads:
            self.grad_dict[k[1]] = k[0]
        self.gradients = []
        for i in range(len(self.hidden_dims) + 1):
            self.gradients.append(self.grad_dict[self.trainable["W{}".format(i)]])
            self.gradients.append(self.grad_dict[self.trainable["b{}".format(i)]])

        self.policy_gradients = [tf.placeholder(tf.float32, shape=[self.state_dim, self.hidden_dims[0]]), tf.placeholder(tf.float32, shape=(self.hidden_dims[0]))]
        for i in np.arange(1, len(self.hidden_dims)):
            self.policy_gradients.append(tf.placeholder(tf.float32, shape=[self.hidden_dims[i - 1], self.hidden_dims[i]]))
            self.policy_gradients.append(tf.placeholder(tf.float32, shape=[self.hidden_dims[i]]))
        self.policy_gradients.append(tf.placeholder(tf.float32, shape=[self.hidden_dims[-1], self.num_actions]))
        self.policy_gradients.append(tf.placeholder(tf.float32, shape=[self.num_actions]))

        self.policy_grad_tuples = []
        for i in range(len(self.hidden_dims) + 1):
            self.policy_grad_tuples.append((self.policy_gradients[2 * i], self.trainable["W{}".format(i)]))
            self.policy_grad_tuples.append((self.policy_gradients[2 * i + 1], self.trainable["b{}".format(i)]))

        self.train_op = self.optimizer.apply_gradients(self.policy_grad_tuples)

        # for index in range(self.batch_size):
        #     self.policy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logprobs[index], labels=self.rollout_actions[index]))
        #     self.loss = self.policy_loss + self.l2_weight * self.l2_loss
        #     self.gradients.append(self.optimizer.compute_gradients(self.loss))
        #
        # self.policy_gradient = {}
        # for i, gradient in enumerate(self.gradients):
        #     for grad, var in gradient:
        #         if var not in self.policy_gradient:
        #             self.policy_gradient[var] = grad * self.discounted_rewards[i]
        #         else:
        #             self.policy_gradient[var] += grad * self.discounted_rewards[i]
        #
        # self.policy_gradient = [(self.policy_gradient[k], k) for k in self.policy_gradient]
        # self.train_op = self.optimizer.apply_gradients(self.policy_gradient)

    def sample_action(self, state):
        if np.random.rand() <= self.exploration_param:
            sampled = np.random.randint(self.num_actions)
        else:
            probs = self.session.run(self.probs, feed_dict={self.inputs: np.expand_dims(state, axis=0)})[0]
            sampled = np.random.choice(self.num_actions, 1, p=probs)[0]
        return sampled

    def train(self):
        self.session.run(tf.global_variables_initializer())

        self.last_rewards = 0
        for epoch in range(self.epochs):
            if epoch % 10 == 0:
                print("Epoch {}, avg reward for last 100 is: {}".format(epoch, self.last_rewards / 100))
                self.last_rewards = 0
            inputs = []
            rollout_actions = []
            discounted_rewards = []

            for batch_num in range(self.batch_size):
                batch_rewards = []

                state = self.env.reset()
                done = False
                max_steps = 200

                num_steps = 0
                while not done and num_steps < max_steps:
                    inputs.append(state)
                    self.env.render()

                    action = self.sample_action(state)
                    rollout_actions.append(action)

                    state, reward, done, _ = self.env.step(action)
                    if not done:
                        batch_rewards.append(0.1)
                    else:
                        batch_rewards.append(-10)

                    num_steps += 1

                self.last_rewards += num_steps
                print(num_steps)

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

            # accumulate gradients for batch and weight by reward
            policy_gradients = [0] * 2 * (len(self.hidden_dims) + 1)

            for i in range(len(inputs)):
                grads = self.session.run(self.gradients, feed_dict={
                    self.inputs: np.array([inputs[i]], dtype=np.float32),
                    self.rollout_actions: np.array([rollout_actions[i]], dtype=np.int32),
                    self.discounted_rewards: np.array([discounted_rewards[i]], dtype=np.float32)
                })
                for j, grad in enumerate(grads):
                    policy_gradients[j] += discounted_rewards[i] * grad

            _ = self.session.run(self.train_op, feed_dict={
                k: v for (k, v) in zip(self.policy_gradients, policy_gradients)
            })

            # anneal exploration parameter
            self.exploration_param = max(self.exploration_param - self.exploration_decay, 0)

        print("Finished training, final reward is: {}".format(batch_rewards[-1]))

if __name__ == "__main__":
    test = CARTPOLE_RL()
    test.train()

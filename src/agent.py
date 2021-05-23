from itertools import count
import numpy as np
import os
import tensorflow.compat.v1 as tf
import gd_utils
from random import sample as external_sample, choice
import time
import cv2
from collections import namedtuple
tf.disable_v2_behavior()

Transition = namedtuple("Transition", ["state", "action", "rewards", "next_state", "done"])


class RelatedMemoryReplay:
    # TODO Add normalization statistics (mean, max, min)
    def __init__(self, memory_len=75_000, feature_shape=128, file_name='replay_memory'):
        self.feature_vectors = np.zeros(shape=(memory_len, feature_shape), dtype=np.float32)
        self.memory = []
        if os.path.exists(os.path.join(os.path.curdir, f"{file_name}.dat")):
            # self.load(file_name)
            self.BASE_MEMORY = len(self.memory)
        else:
            self.BASE_MEMORY = 0
        self.index = 0
        self.MAX_ELEMENTS_VOLATILE = memory_len - self.BASE_MEMORY
        self.v = tf.placeholder(tf.float32, shape=feature_shape)
        self.vs = tf.placeholder(tf.float32, shape=[5_000, feature_shape])
        sample_mean, sample_max, sample_min = tf.reduce_mean(self.vs, 0), tf.reduce_max(self.vs, 0),\
                                              tf.reduce_min(self.vs, 0)
        self.v = tf.divide(tf.subtract(self.v, sample_mean),
                           tf.add(tf.subtract(sample_max, sample_min), tf.constant(1e-7)))
        self.vs = tf.divide(tf.subtract(self.vs, sample_mean),
                            tf.add(tf.subtract(sample_max, sample_min), tf.constant(1e-7)))
        normalize_v = tf.nn.l2_normalize(self.v, 0)
        normalize_vs = tf.nn.l2_normalize(self.vs, 1)
        self.cos_similarity = tf.math.abs(tf.reduce_sum(tf.multiply(normalize_v, normalize_vs), axis=1))
        # self.manh_distance = tf.reduce_sum(tf.abs(tf.subtract(self.v, self.vs)), axis=1)

        # self.feature_vectors_mean = 0
        # self.feature_vectors_max = 0
        # self.feature_vectors_min = 0

    def __len__(self):
        return len(self.memory)

    # TODO Add feature vector normalization before assignment
    def add(self, feature_vec, transition):
        self.index %= self.MAX_ELEMENTS_VOLATILE
        index = self.index + self.BASE_MEMORY
        if len(self.memory) < self.MAX_ELEMENTS_VOLATILE + self.BASE_MEMORY:
            self.memory.append(transition)
        else:
            self.memory[index] = transition
        self.feature_vectors[index] = feature_vec
        self.index += 1

    # TODO create private method to normalize input feature vector
    # def _normalize(self, feature_vec):
    #     self.feature_vectors_max = np.maximum(self.feature_vectors_max, feature_vec)
    #     self.feature_vectors_min = np.minimum(self.feature_vectors_min, feature_vec)
    #     self.feature_vectors_mean += 1.0 / 10_000 * (feature_vec - self.feature_vectors_mean)
    #     feature_vec = (feature_vec - self.feature_vectors_mean) /\
    #                   (self.feature_vectors_max - self.feature_vectors_min + 1e-7)
    #     return feature_vec

    # TODO Add input vector normalization before similarity calculation
    def sample_like(self, feature_vec, batch_size: int, sess, alpha=0.6):
        sample_ixs = np.arange(self.MAX_ELEMENTS_VOLATILE + self.BASE_MEMORY)[:len(self.memory)]
        sample_ixs = np.random.choice(sample_ixs, 5_000)
        feature_vecs = np.array([self.feature_vectors[ix] for ix in sample_ixs])
        similarities = sess.run(self.cos_similarity, feed_dict={self.v: feature_vec, self.vs: feature_vecs})
        similarities = similarities ** alpha
        similarities /= np.sum(similarities)
        sample_ixs = np.random.choice(sample_ixs, batch_size, p=similarities)
        return [self.memory[i] for i in sample_ixs], sample_ixs

    # TODO Add feature vector normalization before assignment
    def update_feature_vecs(self, new_feature_vecs, ixs):
        for i, w in zip(ixs, new_feature_vecs):
            self.feature_vectors[i] = w

    # TODO create initial norm method
    # def initial_norm(self):
    #     self.feature_vectors_mean = self.feature_vectors[:len(self.memory)].mean(axis=0)
    #     self.feature_vectors_max = self.feature_vectors[:len(self.memory)].max(axis=0)
    #     self.feature_vectors_min = self.feature_vectors[:len(self.memory)].min(axis=0)
    #     self.feature_vectors[:len(self.memory)] = \
    #         (self.feature_vectors[:len(self.memory)] - self.feature_vectors_mean) / \
    #         (self.feature_vectors_max - self.feature_vectors_min + 1e-7)

    def sample_random(self, batch_size: int):
        ixs_random = external_sample(range(len(self.memory)), batch_size)
        return [self.memory[i] for i in ixs_random]

    # def save(self, file_name="replay_memory"):
    #     with shelve.open(file_name) as sh:
    #         sh["memory"] = self.memory
    #         sh["feature_vectors"] = self.feature_vectors
    #
    # def load(self, file_name="replay_memory"):
    #     with shelve.open(file_name) as sh:
    #         self.memory = sh["memory"]
    #         loaded_vectors = sh["feature_vectors"]
    #     self.feature_vectors[:len(self.memory)] = loaded_vectors[:len(self.memory)]


class Estimator:
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """
        # Model architecture and placeholders
        # Our input are 4 grayscale frames of shape 84, 84 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        # Learning rate to use
        self.lr = tf.placeholder(tf.float32, shape=[])

        # Three convolutional layers
        self.conv1_layer = tf.keras.layers.Conv2D(
            32, 8, 4, activation=tf.nn.relu,
            kernel_initializer='glorot_normal',
            # kernel_regularizer=tf.keras.regularizers.l1(0.001),
            # activity_regularizer=tf.keras.regularizers.l1(0.001),
            # bias_regularizer=tf.keras.regularizers.l1(0.001)
        )
        self.conv2_layer = tf.keras.layers.Conv2D(
            64, 4, 2, activation=tf.nn.relu,
            kernel_initializer='glorot_normal',
            # kernel_regularizer=tf.keras.regularizers.l1(0.001),#000001
            # activity_regularizer=tf.keras.regularizers.l1(0.001),
            # bias_regularizer=tf.keras.regularizers.l1(0.001)
        )
        self.conv3_layer = tf.keras.layers.Conv2D(
            64, 3, 1, activation=tf.nn.relu,
            kernel_initializer='glorot_normal',
            # kernel_regularizer=tf.keras.regularizers.l1(0.001),
            # activity_regularizer=tf.keras.regularizers.l1(0.001),
            # bias_regularizer=tf.keras.regularizers.l1(0.001)
        )

        # Flatten convolutional output
        flattened_layer = tf.keras.layers.Flatten()
        dr1 = tf.keras.layers.Dropout(0.3)
        # Fully connected layers
        self.fc1_layer = tf.keras.layers.Dense(
            512, activation=tf.nn.relu,
            kernel_initializer='glorot_normal',
            # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            # activity_regularizer=tf.keras.regularizers.l1(0.001),
            # bias_regularizer=tf.keras.regularizers.l1(0.001)
        )
        dr2 = tf.keras.layers.Dropout(0.3)
        self.fc2_layer = tf.keras.layers.Dense(
            128, activation=tf.nn.relu,
            kernel_initializer='glorot_normal',
            # kernel_regularizer=tf.keras.regularizers.l1(0.001),
            # activity_regularizer=tf.keras.regularizers.l1(0.001),
            # bias_regularizer=tf.keras.regularizers.l1(0.001)
        )
        self.predictions_layer = tf.keras.layers.Dense(
            len(VALID_ACTIONS), activation=None,
            kernel_initializer='glorot_normal',
            # kernel_regularizer=tf.keras.regularizers.l1(0.001),
            # activity_regularizer=tf.keras.regularizers.l1(0.001),
            # bias_regularizer=tf.keras.regularizers.l1(0.001)
        )

        # Receive input
        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Propagate through the network
        conv1_out = self.conv1_layer(X)
        conv2_out = self.conv2_layer(conv1_out)
        conv3_out = self.conv3_layer(conv2_out)
        flattened_out = dr1(flattened_layer(conv3_out), training=True)
        self.fc1_out = dr2(self.fc1_layer(flattened_out), training=True)
        self.fc2_out = self.fc2_layer(self.fc1_out)
        self.predictions_out = self.predictions_layer(self.fc2_out)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions_out)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions_out, [-1]), gather_indices)

        # Calculate the loss
        td_errors = tf.math.abs(tf.math.subtract(self.y_pl, self.action_predictions))
        # self.feature_vec = tf.divide(tf.subtract(self.feature_vec, self.feature_vectors_mean),
        #                              tf.add(tf.subtract(self.feature_vectors_max, self.feature_vectors_min), 1e-7))
        # self.fc1_out = tf.divide(tf.subtract(self.fc1_out, self.feature_vectors_mean),
        #                          tf.add(tf.subtract(self.feature_vectors_max, self.feature_vectors_min), 1e-7))
        # distances = tf.reduce_sum(tf.abs(tf.subtract(self.feature_vec, self.fc1_out)), axis=1)
        # similarities = tf.math.pow(tf.subtract(tf.reduce_sum(distances), distances), 0.6)
        # probs = tf.divide(similarities, tf.add(tf.reduce_sum(similarities), 1e-7))
        self.losses = tf.math.square(td_errors)
        self.loss = tf.reduce_mean(self.losses)
        # self.loss = tf.math.add_n([tf.reduce_sum(tf.math.multiply(self.losses, probs)),
        #                            tf.reduce_sum(self.fc1_layer.losses),
        #                            ])
        #             tf.reduce_sum(self.fc2_layer.losses),
        #             tf.reduce_sum(self.predictions_layer.losses),
        #             tf.reduce_sum(conv1_layer.losses),
        #             tf.reduce_sum(conv2_layer.losses),
        #             tf.reduce_sum(conv3_layer.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(self.lr, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # Summaries for Tensorboard

        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions_out),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions_out))
        ])

    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 120, 120, 1]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        predictions, feature_vec = sess.run([self.predictions_out, self.fc2_out], {self.X_pl: np.expand_dims(s, 3)})
        return predictions, np.squeeze(feature_vec)

    def update(self, sess, s, a, y, lr):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 120, 120, 1]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """

        feed_dict = {self.X_pl: np.expand_dims(s, 3), self.y_pl: y, self.actions_pl: a,
                     self.lr: lr}
        summaries, global_step,  _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

    def reset(self):
        self.conv1_layer.kernel.initializer.run(session=sess)
        self.conv1_layer.bias.initializer.run(session=sess)
        self.conv2_layer.kernel.initializer.run(session=sess)
        self.conv2_layer.bias.initializer.run(session=sess)
        self.conv3_layer.kernel.initializer.run(session=sess)
        self.conv3_layer.bias.initializer.run(session=sess)
        self.fc1_layer.kernel.initializer.run(session=sess)
        self.fc1_layer.bias.initializer.run(session=sess)
        self.fc2_layer.kernel.initializer.run(session=sess)
        self.fc2_layer.bias.initializer.run(session=sess)
        self.predictions_layer.kernel.initializer.run(session=sess)
        self.predictions_layer.bias.initializer.run(session=sess)


class ModelParametersCopier:
    """
    Copy model parameters of one estimator to another.
    """

    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)
                
        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op1 = e2_v.assign(e1_v)
            self.update_ops.append(op1)

    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(sess, observation, epsilon, batch_size):
        if batch_size:
            A = np.ones((batch_size, nA), dtype=float) * epsilon / nA
            q_values, feature_vec = estimator.predict(sess, observation)
            best_actions = np.argmax(q_values, axis=1)
            A[np.arange(batch_size), best_actions] += (1.0 - epsilon)
        else:
            A = np.ones(nA, dtype=float) * epsilon / nA
            q_values, feature_vec = estimator.predict(sess, np.expand_dims(observation, 0))
            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
        return A, q_values, feature_vec

    return policy_fn


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=50_000,
                    replay_memory_init_size=10_000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=0.1,
                    epsilon_end=0.0001,
                    epsilon_decay_steps=300_000,
                    batch_size=32):
    """
    N(10)-step Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: Geometry Dash environment emulator
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sample when initializing
          the replay memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value (Not used)
        epsilon_end: The final minimum value of epsilon after decaying is done (Not used)
        epsilon_decay_steps: Number of steps to decay epsilon over (Not used)
        batch_size: Size of batches to sample from the replay memory

    Returns:
        Yields tuple of (Episode loss, Number of parameter updates) at the end of every episode.
    """

    replay_memory = RelatedMemoryReplay(replay_memory_size)
    copier = ModelParametersCopier(q_estimator, target_estimator)

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Get the current time step
    total_t = sess.run(tf.train.get_global_step())

    # The learning rate decay schedule
    lrs = np.linspace(0.000_25, 0.000_025, 500_000)
    # alphas = np.linspace(0., 0.8, 800_000)
    epsilon = 0.2
    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    print("Populating replay memory...")
    n_step = 10
    change = 1
    while len(replay_memory) < replay_memory_init_size:      
        env.retry()
        done = False
        elapsed_time = time.perf_counter()
        state, _, _, fr, _, _ = env.step(0)
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # state = cv2.Canny(state, threshold1=275, threshold2=300)
        # state = np.stack([state] * 4, axis=2)

        rewards_window = []
        states_window = []
        actions_window = []
        feature_vecs = []
        for t in count(start=1):
            # Populate replay memory!
            # state_image = (state[:, :, 0] * 0.4 + state[:, :, 1] * 0.3 + state[:, :, 2] * 0.2 + state[:, :, 3] * 0.1).astype(np.uint8)
            action_probs, _, feature_vec = policy(sess, state, epsilon, None)
            # action_probs = [1 - epsilon, epsilon]
            action = np.random.choice(VALID_ACTIONS, p=action_probs)
            next_frame, reward, done, fr, _, _ = env.step(action)
            # feature_vec = target_estimator.get_feature_vector(sess, state)

            rewards_window.append(reward)
            states_window.append(state)
            actions_window.append(action)
            feature_vecs.append(feature_vec)

            next_frame_g = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
            # image = cv2.Canny(next_frame_g, threshold1=275, threshold2=300)
            next_state = (state * 0.6 + 0.4 * next_frame_g).astype(np.uint8)
            # next_state_image = (next_state[:, :, 0] * 0.4 + next_state[:, :, 1] * 0.3 + next_state[:, :, 2] * 0.2 + next_state[:, :, 3] * 0.1).astype(np.uint8)

            if len(rewards_window) == n_step:
                replay_memory.add(
                    feature_vecs[0],
                    Transition(states_window[0], actions_window[0], rewards_window[:], next_state, done),
                    # replay=True
                )
                states_window.pop(0)
                rewards_window.pop(0)
                actions_window.pop(0)
                feature_vecs.pop(0)

            if done:
                while len(rewards_window) != 0:
                    replay_memory.add(
                        feature_vecs[0],
                        Transition(states_window[0], actions_window[0],
                                   rewards_window[:] + [0 for _ in range(n_step - len(rewards_window))],
                                   next_state, done),
                        # replay=True
                    )
                    states_window.pop(0)
                    rewards_window.pop(0)
                    actions_window.pop(0)
                    feature_vecs.pop(0)
                print("\n" + "-" * 50)
                break
            else:
                state = next_state
                time.sleep(1.0 / 70)

        elapsed_time = time.perf_counter() - elapsed_time
        fps = fr / elapsed_time
        print(f"Replay memory length: {len(replay_memory)}")
        print(f"Framerate: {fps} | frames: {fr}")

    # replay_memory.initial_norm()
    print("Training started!")
    n_step_discount = np.array([discount_factor ** i for i in range(n_step)])
    for i_episode in range(num_episodes):
        # Reset the environment
        env.retry()
        ep_reward = 0
        elapsed_time = time.perf_counter()
        state, _, _, fr, reached, attempts = env.step(0)
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # state = cv2.Canny(state, threshold1=275, threshold2=300)
        # state = np.stack([state] * 4, axis=2)
        frames_to_write = [env.record_frame]

        loss = 0
        update_num = 0.001
        done = False
        rewards_window = []
        states_window = []
        actions_window = []
        feature_vecs = []

        # One step in the environment
        for t in count(start=1):
            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                env.pause()
                copier.make(sess)
                # q_estimator.reset()
                env.unpause()

            # Take a step in the environment
            action_probs, action_values, feature_vec = policy(sess, state, 0, None)
            action = np.random.choice(VALID_ACTIONS, p=action_probs)
            next_frame, reward, done, fr, reached, attempts = env.step(action)
            # feature_vec = target_estimator.get_feature_vector(sess, state)

            rewards_window.append(reward)
            states_window.append(state)
            actions_window.append(action)
            feature_vecs.append(feature_vec)

            if (i_episode + 1) % 100 == 0:
                frames_to_write.append(env.record_frame)
            print(f"\rAction values: {action_values}, total_t: {total_t}", end="")

            # Save transition to replay memory
            next_frame_g = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
            # image = cv2.Canny(next_frame_g, threshold1=275, threshold2=300)
            next_state = (state * 0.6 + 0.4 * next_frame_g).astype(np.uint8)
            # next_state_image = (next_state[:, :, 0] * 0.4 + next_state[:, :, 1] * 0.3 + next_state[:, :, 2] * 0.2 + next_state[:, :, 3] * 0.1).astype(np.uint8)

            if len(rewards_window) == n_step:
                replay_memory.add(
                    feature_vecs[0],
                    Transition(states_window[0], actions_window[0], rewards_window[:], next_state, done),
                )
                states_window.pop(0)
                rewards_window.pop(0)
                actions_window.pop(0)
                feature_vecs.pop(0)
            ep_reward += reward

            # Sample related transitions from memory replay
            samples_rlt, ixs = replay_memory.sample_like(feature_vec, batch_size, sess, alpha=0.7)
            # samples_rlt = replay_memory.sample_random(batch_size)
            # samples_rlt = replay_memory.sample_random(batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples_rlt))

            # This is where Double Q-Learning comes in!
            q_values_next, new_feature_vecs = q_estimator.predict(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1).astype(np.int32)
            q_values_next_target, _ = target_estimator.predict(sess, next_states_batch)
            td_targets = (np.dot(reward_batch, n_step_discount) + (1 - done_batch) *
                          discount_factor ** n_step * q_values_next_target[np.arange(len(best_actions)), best_actions]).astype(np.float32)

            # Learning rate decays until 500k parameter updates
            lr = lrs[min(499_999, total_t)]
            # lr = 0.000_25
            batch_loss = q_estimator.update(sess, states_batch, action_batch, td_targets, lr)
            loss += batch_loss
            update_num += 1
            total_t += 1
            replay_memory.update_feature_vecs(new_feature_vecs, ixs)

            if done:
                while len(rewards_window) != 0:
                    replay_memory.add(
                                      feature_vecs[0],
                                      Transition(states_window[0], actions_window[0],
                                                 rewards_window[:] + [0 for _ in range(n_step - len(rewards_window))],
                                                 next_state, done),
                                      )
                    states_window.pop(0)
                    rewards_window.pop(0)
                    actions_window.pop(0)
                    feature_vecs.pop(0)
                print("\n" + "-" * 50)
                break
            else:
                state = next_state

        loss /= update_num
        elapsed_time = time.perf_counter() - elapsed_time
        fps = fr / elapsed_time
        print(f"\nEpisode's framerate: {fps}")
        print(f"Replay memory len : {len(replay_memory)}")

        if (i_episode + 1) % 25 == 0:
            env.pause()
            # Save the current checkpoint
            saver.save(tf.get_default_session(), checkpoint_path)
            # Write episode to video every 100 episodes:
            if (i_episode + 1) % 100 == 0:
                height, width, _ = frames_to_write[0].shape
                output = cv2.VideoWriter(os.path.join(videos_dir, f'episode{i_episode + 1}.avi'),
                                         cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
                for frame in frames_to_write:
                    output.write(frame.astype('uint8'))
                output.release()
            env.unpause()

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        if done and reached:
            episode_summary.value.add(simple_value=attempts, tag="episode/attempts")
        episode_summary.value.add(simple_value=ep_reward, tag="episode/reward")
        episode_summary.value.add(simple_value=fr, tag="episode/length")
        episode_summary.value.add(simple_value=loss, tag="episode/loss")
        q_estimator.summary_writer.add_summary(episode_summary, i_episode + 1)
        q_estimator.summary_writer.flush()
        yield total_t, loss

    return 0


env = gd_utils.GDenv((800, 600), mode="practice")
VALID_ACTIONS = [0, 1]
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/Geometry Dash")
videos_dir = os.path.abspath("./videos")

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# Run it!
with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i, (t, ep_reward) in enumerate(deep_q_learning(sess,
                                                       env,
                                                       q_estimator=q_estimator,
                                                       target_estimator=target_estimator,
                                                       experiment_dir=experiment_dir,
                                                       num_episodes=20_000,
                                                       replay_memory_size=100_000,
                                                       replay_memory_init_size=10_000,
                                                       update_target_estimator_every=10_000,
                                                       discount_factor=0.9,
                                                       batch_size=32)
                                       ):
        print(f"Episode's #{i + 1} loss: {ep_reward}")
print("Training done!")

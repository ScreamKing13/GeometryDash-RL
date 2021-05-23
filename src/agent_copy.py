from itertools import count
import numpy as np
import os
import tensorflow as tf
import gd_utils
from random import sample as external_sample
import time
import cv2
from collections import namedtuple
import shelve
import matplotlib.pyplot as plt
from pylab import *
import keyboard

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
Transition = namedtuple("Transition", ["state", "action", "rewards", "next_state", "done"])


class RelatedMemoryReplay:
    def __init__(self, feature_shape, memory_len=75_000, file_name='replay_memory'):
        self.feature_vectors = np.zeros(shape=(memory_len, feature_shape), dtype=np.float32)
        self.priorities = np.empty(shape=(memory_len), dtype=np.float32)
        self.memory = []
        self.max_priority = 1.0
        if os.path.exists(os.path.join(os.path.curdir, f"{file_name}.dat")):
            self.load(file_name)
            self.BASE_MEMORY = len(self.memory)
            self.priorities[:self.BASE_MEMORY] = np.array([self.max_priority for _ in range(self.BASE_MEMORY)])
        else:
            self.BASE_MEMORY = 0
        # Normalize the feature vectors
        # if self.BASE_MEMORY != 0:
        #     self.initial_norm()
        self.index = 0
        self.MAX_ELEMENTS_VOLATILE = memory_len - self.BASE_MEMORY
        self.v = tf.placeholder(tf.float32, shape=feature_shape)
        self.vs = tf.placeholder(tf.float32, shape=self.feature_vectors.shape)
        normalize_v = tf.nn.l2_normalize(self.v, 0)
        normalize_vs = tf.nn.l2_normalize(self.vs, 1)
        self.cos_similarity = tf.math.divide(tf.math.add(1.0, tf.reduce_sum(tf.multiply(normalize_v, normalize_vs), axis=1)), 2.0)
        # self.angle_similarity = tf.math.subtract(1.0, tf.math.divide(tf.math.acos(self.cos_similarity), np.pi))

    def __len__(self):
        return len(self.memory)

    def add(self, feature_vec, transition):
        self.index %= self.MAX_ELEMENTS_VOLATILE
        index = self.index + self.BASE_MEMORY
        if len(self.memory) < self.MAX_ELEMENTS_VOLATILE + self.BASE_MEMORY:
            self.memory.append(transition)
        else:
            self.memory[index] = transition
        self.feature_vectors[index] = feature_vec
        self.priorities[index] = self.max_priority
        self.index += 1

    def sample_like(self, feature_vec, batch_size: int, sess, alpha=0.6):
        similarities = sess.run(self.cos_similarity,
                                feed_dict={self.v: feature_vec, self.vs: self.feature_vectors})
        ixs = np.arange(self.MAX_ELEMENTS_VOLATILE + self.BASE_MEMORY)[:len(self.memory)]
        similarities = similarities[:len(self.memory)]
        similarities = similarities ** alpha
        # priorities = np.abs(self.priorities[:len(self.memory)]) + 1e-5
        similarities /= np.sum(similarities)
        # priorities /= np.sum(priorities)
        # final_metric = alpha * similarities + (1 - alpha) * priorities
        ixs = np.random.choice(ixs, batch_size, p=similarities)
        return [self.memory[i] for i in ixs], ixs

    def rebase(self, estimator, sess):
        for i, entry in enumerate(self.memory):
            self.feature_vectors[i] = estimator.get_feature_vector(sess, entry.state)

    # def update_priotities(self, new_priorities, ixs):
    #     for i, p in zip(ixs, new_priorities):
    #         self.priorities[i] = p
    #         self.max_priority = max(self.max_priority, abs(p))

    # def normalize(self, feature_vec):
    #     self.feature_vectors_max = np.maximum(self.feature_vectors_max, feature_vec)
    #     self.feature_vectors_min = np.minimum(self.feature_vectors_min, feature_vec)
    #     self.feature_vectors_mean += 1 / (self.feature_vectors.shape[0] / 10) * \
    #                                  (feature_vec - self.feature_vectors_mean)
    #     feature_vec = (feature_vec - self.feature_vectors_mean) / \
    #                            (self.feature_vectors_max - self.feature_vectors_min)
    #     return feature_vec

    # def initial_norm(self):
    #     self.feature_vectors_mean = self.feature_vectors[:len(self.memory)].mean(axis=0)
    #     self.feature_vectors_max = self.feature_vectors[:len(self.memory)].max(axis=0)
    #     self.feature_vectors_min = self.feature_vectors[:len(self.memory)].min(axis=0)
    #     self.feature_vectors[:len(self.memory)] = (self.feature_vectors[
    #                                                :len(self.memory)] - self.feature_vectors_mean) / \
    #                                               (self.feature_vectors_max - self.feature_vectors_min)

    def sample_random(self, batch_size: int):
        ixs_random = external_sample(range(len(self.memory)), batch_size)
        return [self.memory[i] for i in ixs_random]

    def save(self, file_name="replay_memory"):
        with shelve.open(file_name) as sh:
            sh["memory"] = self.memory
            sh["feature_vectors"] = self.feature_vectors

    def load(self, file_name="replay_memory"):
        with shelve.open(file_name) as sh:
            self.memory = sh["memory"]
            loaded_vectors = sh["feature_vectors"]
        self.feature_vectors[:len(self.memory)] = loaded_vectors[:len(self.memory)]


# class FrameProcessor:
#     def __init__(self, input_shape):
#         i = tf.keras.layers.Input(input_shape, dtype=tf.uint8, batch_size=1)
#         x = tf.keras.applications.vgg16.preprocess_input(tf.cast(i, tf.float32))
#         core = tf.keras.applications.VGG16(include_top=False)
#         core = tf.keras.Sequential(core.layers[:7])
#         x = core(x)
#         out = tf.keras.layers.GlobalAveragePooling2D()(x)
#         self.model = tf.keras.Model(inputs=[i], outputs=[out])
#         self.model.trainable = False
#
#     def process(self, frame):
#         return self.model.predict(np.expand_dims(frame, axis=0))[0]


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

        # Placeholders for our input
        # Our input are 4 grayscale frames of shape 84, 84 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.training = tf.compat.v1.placeholder_with_default(True, shape=[])

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.keras.layers.Conv2D(32, 8, 4, activation=tf.nn.relu, kernel_initializer='glorot_normal')(X)
        # sp_dr1 = tf.keras.layers.SpatialDropout2D(0.2)(conv1, training=self.training)
        conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu, kernel_initializer='glorot_normal')(conv1)
        # sp_dr2 = tf.keras.layers.SpatialDropout2D(0.2)(conv2, training=self.training)
        conv3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu, kernel_initializer='glorot_normal')(conv2)
        # sp_dr3 = tf.keras.layers.SpatialDropout2D(0.2)(conv3, training=self.training)
        # Fully connected layers
        flattened = tf.keras.layers.Flatten()(conv3)
        self.fc1_l = tf.keras.layers.Dense(512, activation=tf.nn.relu,
                                         kernel_initializer='glorot_normal',
                                         # kernel_regularizer=tf.keras.regularizers.l2(0.0001)
                                         )
        self.fc1_o  = self.fc1_l(flattened)
        self.bn1 = tf.keras.layers.BatchNormalization()(self.fc1_o, training=self.training)
        self.fc2_l = tf.keras.layers.Dense(128, activation=tf.nn.relu,
                                         kernel_initializer='glorot_normal',
                                         # kernel_regularizer=tf.keras.regularizers.l2(0.0001)
                                         )
        self.fc2_o  = self.fc2_l(self.bn1)
        self.bn2 = tf.keras.layers.BatchNormalization()(self.fc2_o, training=self.training)
        # self.features = tf.keras.layers.BatchNormalization()(self.fc1)
        # dr2 = tf.keras.layers.Dropout(0.4)(self.fc1, training=self.training)
        self.predictions_l = tf.keras.layers.Dense(len(VALID_ACTIONS), activation=None,
                                                 kernel_initializer='glorot_normal',
                                                 # kernel_regularizer=tf.keras.regularizers.l2(0.0001)
                                                 )
        self.predictions = self.predictions_l(self.bn2)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.td_errors = tf.subtract(self.y_pl, self.action_predictions)
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        # self.loss = tf.math.add_n([tf.reduce_mean(self.losses), self.fc1_l.losses[0],
        #                            self.fc2_l.losses[0], self.predictions_l.losses[0]])
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(self.lr, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # Summaries for Tensorboard

        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s, training=True):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 120, 120, 1]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return sess.run(self.predictions, {self.X_pl: s, self.training: training})

    def get_feature_vector(self, sess, s):
        feature_vec = sess.run(self.bn2, {self.X_pl: np.expand_dims(s, 0)})
        return np.squeeze(feature_vec)

    def calc_loss(self, sess, s, a, y):
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        loss = sess.run(
            [self.loss],
            feed_dict)
        return loss[0]

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

        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a, self.lr: lr}
        summaries, global_step,  _, loss, td_errors = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss, self.td_errors],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss, td_errors


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
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

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

    def policy_fn(sess, observation, epsilon, batch_size, training=True):
        if batch_size:
            A = np.ones((batch_size, nA), dtype=float) * epsilon / nA
            q_values = estimator.predict(sess, observation, training)
            best_actions = np.argmax(q_values, axis=1)
            A[np.arange(batch_size), best_actions] += (1.0 - epsilon)
        else:
            A = np.ones(nA, dtype=float) * epsilon / nA
            q_values = estimator.predict(sess, np.expand_dims(observation, 0), training)[0]
            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
        return A, q_values

    return policy_fn


def random_mini_batches(memory, mini_batch_size=64):
    """
    Creates a list of random minibatches from X

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = len(memory)  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    np.random.shuffle(memory)

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(np.floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batches.append([memory[j] for j in range(mini_batch_size * k, mini_batch_size * (k + 1))])
    k = num_complete_minibatches - 1
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batches.append([memory[j] for j in range(mini_batch_size * (k + 1), m)])

    return mini_batches


def put_to_mem(mem, fr, a_vec, a_next, transition):
    mem[(fr, tuple(a_vec), a_next)] = transition


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
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sample when initializing
          the replay memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # processor = FrameProcessor((84, 84, 3))
    replay_memory = RelatedMemoryReplay(128, replay_memory_size)
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

    # The epsilon decay schedule
    lrs = np.linspace(0.00025, 0.000025, 500_000)
    # max_attempt_length = 0
    # attempt_discount = 0.7

    epsilon = 0.
    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    print("Populating replay memory...")
    n_step = 10
    # index = 0
    # frames_per_episode = np.array([40 for _ in range(3)])
    # checkpoint_every = 2
    # checkpointed = False
    while len(replay_memory) < replay_memory_init_size:
        env.retry()
        done = False
        elapsed_time = time.perf_counter()
        state, _, _, fr, _, _ = env.step(0)
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.Canny(state, threshold1=275, threshold2=300)
        state = np.stack([state] * 4, axis=2)

        rewards_window = []
        states_window = []
        actions_window = []
        feature_vecs = []
        for t in count(start=1):
            # Populate replay memory!
            action_probs, _ = policy(sess, state, epsilon, None, training=False)
            action = np.random.choice(VALID_ACTIONS, p=action_probs)
            # action = 1 if keyboard.is_pressed('W') else 0
            next_frame, reward, done, fr, _, _ = env.step(action)
            feature_vec = target_estimator.get_feature_vector(sess, state)
            rewards_window.append(reward)
            states_window.append(state)
            actions_window.append(action)
            feature_vecs.append(feature_vec)
            next_frame_g = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
            image = cv2.Canny(next_frame_g, threshold1=275, threshold2=300)
            next_state = np.append(state[:, :, 1:], np.expand_dims(image, axis=2), axis=2)
            # feature_vec = processor.process(next_frame)
            if len(rewards_window) == n_step:
                replay_memory.add(feature_vecs[0],
                                  Transition(states_window[0], actions_window[0], rewards_window[:], next_state, done),
                                  )
                states_window.pop(0)
                rewards_window.pop(0)
                actions_window.pop(0)
                feature_vecs.pop(0)
            # if int(time.perf_counter() - elapsed_time + 1) % (checkpoint_every) == 0:
            #     if not checkpointed:
            #         env.checkpoint()
            #         checkpoint_every = 2
            #         checkpointed = True
            # else:
            #     checkpointed = False
            if done:
                while len(rewards_window) != 0:
                    replay_memory.add(feature_vecs[0],
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
                # time.sleep(1.0 / 40)
                time.sleep(1.0 / 70)

        # max_attempt_length = max(max_attempt_length, fr)
        elapsed_time = time.perf_counter() - elapsed_time
        fps =  fr / elapsed_time
        # index %= 3
        # frames_per_episode[index] = fr
        # index += 1
        # if np.all(frames_per_episode < 40):
        #     env.uncheckpoint()
            # checkpoint_every -= 1
            # frames_per_episode = np.array([40 for _ in range(3)])
        print(f"Replay memory length: {len(replay_memory)}")
        print(f"Framerate: {fps} | frames: {fr}")
        # print(f"Current max attempt length: {max_attempt_length}, fr: {fr}")

    # print("Memory normalization initialized.")
    # replay_memory.initial_norm()

    # min_loss = 0.01
    print("Training started!")
    # index = 0
    # frames_per_episode = np.array([40 for _ in range(3)])
    # checkpoint_every = 3
    # checkpointed = False
    n_step_discount = np.array([discount_factor ** i for i in range(n_step)])
    for i_episode in range(num_episodes):
        ep_reward = 0
        # Reset the environment
        env.retry()
        elapsed_time = time.perf_counter()
        state, _, _, fr, reached, attempts = env.step(0)
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.Canny(state, threshold1=275, threshold2=300)
        state = np.stack([state] * 4, axis=2)
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

            # Epsilon for this time step
            # epsilon = 0.25
            # eps = 0.1 if fr >= max_attempt_length else 0
            # Maybe update the target estimator
            if (total_t + 1) % update_target_estimator_every == 0:
                env.pause()
                copier.make(sess)
                # replay_memory.rebase(target_estimator, sess)
                env.unpause()

            # Take a step in the environment
            action_probs, action_values = policy(sess, state, 0, None)
            action = np.random.choice(VALID_ACTIONS, p=action_probs)
            next_frame, reward, done, fr, reached, attempts = env.step(action)
            feature_vec = target_estimator.get_feature_vector(sess, state)
            rewards_window.append(reward)
            states_window.append(state)
            actions_window.append(action)
            feature_vecs.append(feature_vec)
            if (i_episode + 1) % 100 == 0:
                frames_to_write.append(env.record_frame)
            print(f"\rAction values: {action_values}, total_t: {total_t}", end="")

            # Save transition to replay memory
            next_frame_g = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
            image = cv2.Canny(next_frame_g, threshold1=275, threshold2=300)
            next_state = np.append(state[:, :, 1:], np.expand_dims(image, axis=2), axis=2)
            # feature_vec = processor.process(next_frame)
            # feature_vec = replay_memory.normalize(feature_vec)
            if len(rewards_window) == n_step:
                replay_memory.add(feature_vecs[0],
                                  Transition(states_window[0], actions_window[0], rewards_window[:], next_state,
                                             done),
                                  )
                states_window.pop(0)
                rewards_window.pop(0)
                actions_window.pop(0)
                feature_vecs.pop(0)
            ep_reward += reward

            # if int(time.perf_counter() - elapsed_time + 1) % (checkpoint_every) == 0:
            #     if not checkpointed:
            #         env.checkpoint()
            #         checkpoint_every = 3
            #         checkpointed = True
            # else:
            #     checkpointed = False

            # Sample a minibatch from the replay memory
            # if np.random.rand() < epsilon:
            #     samples_rnd, ixs = replay_memory.sample_like(feature_vec, batch_size, sess, alpha=0)
            #     states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples_rnd))
            # else:
            samples_rlt, ixs = replay_memory.sample_like(feature_vec, batch_size, sess, alpha=0.6)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples_rlt))
            # samples = replay_memory.sample_random(batch_size)
            # states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            # time.sleep(1/40)

            # This is where Double Q-Learning comes in!
            q_values_next = q_estimator.predict(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            td_targets = np.dot(reward_batch, n_step_discount) + (1 - done_batch) * \
                         discount_factor ** n_step * q_values_next_target[np.arange(len(best_actions)), best_actions]

            lr = lrs[min(499_999, total_t)]
            batch_loss, _ = q_estimator.update(sess, states_batch, action_batch, td_targets, lr)
            loss += batch_loss
            # replay_memory.update_priotities(td_errors, ixs)

            update_num += 1
            total_t += 1
            if done:
                while len(rewards_window) != 0:
                    replay_memory.add(feature_vecs[0],
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
        # index %= 3
        # frames_per_episode[index] = fr
        # index += 1
        # if np.all(frames_per_episode < 40):
        #     env.uncheckpoint()
            # checkpoint_every -= 1
            # frames_per_episode = np.array([40 for _ in range(3)])
        print(f"\nEpisode's framerate: {fps}")
        print(f"Replay memory len : {len(replay_memory)}")
        # print(f"Current max attempt length: {max_attempt_length}, fr: {fr}")
        # max_attempt_length = fr if fr > max_attempt_length else int(max_attempt_length * attempt_discount)

        if (i_episode + 1) % 25 == 0:
            env.pause()
            # Save the current checkpoint
            saver.save(tf.get_default_session(), checkpoint_path)
            if (i_episode + 1) % 100 == 0:
                # Write episode to video:
                height, width, _ = frames_to_write[0].shape
                output = cv2.VideoWriter(os.path.join(videos_dir, f'episode{i_episode + 1}.avi'),
                                         cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
                for frame in frames_to_write:
                    output.write(frame.astype('uint8'))
                output.release()
            env.unpause()

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        # episode_summary.value.add(simple_value=eps, tag="episode/epsilon")
        if done and reached:
            episode_summary.value.add(simple_value=attempts, tag="episode/attempts")
        episode_summary.value.add(simple_value=ep_reward, tag="episode/reward")
        episode_summary.value.add(simple_value=fr, tag="episode/length")
        episode_summary.value.add(simple_value=loss, tag="episode/loss")
        q_estimator.summary_writer.add_summary(episode_summary, i_episode)
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
                                                       replay_memory_size=90_000,
                                                       replay_memory_init_size=105_000,
                                                       update_target_estimator_every=10_000,
                                                       discount_factor=0.9,
                                                       batch_size=32)
                                       ):
        print(f"Episode's #{i + 1} loss: {ep_reward}")
print("Training done!")

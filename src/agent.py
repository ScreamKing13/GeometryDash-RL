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

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class RelatedMemoryReplay:
    def __init__(self, feature_shape, memory_len=75_000, file_name='replay_memory'):
        self.feature_vectors = np.zeros(shape=(memory_len, feature_shape), dtype=np.float32)
        self.memory = []
        if os.path.exists(os.path.join(os.path.curdir, f"{file_name}.dat")):
            self.load(file_name)
            self.BASE_MEMORY = len(self.memory)
        else:
            self.BASE_MEMORY = 0
        # Normalize the feature vectors
        self.feature_vectors_mean = self.feature_vectors[:self.BASE_MEMORY].mean(axis=0)
        self.feature_vectors_max = self.feature_vectors[:self.BASE_MEMORY].max(axis=0)
        self.feature_vectors_min = self.feature_vectors[:self.BASE_MEMORY].min(axis=0)
        self.feature_vectors[:self.BASE_MEMORY] = (self.feature_vectors[:self.BASE_MEMORY] - self.feature_vectors_mean) / \
                                                  (self.feature_vectors_max - self.feature_vectors_min)
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
        self.index += 1

    def sample_like(self, feature_vec, batch_size: int, sess, alpha=0.7):
        similarities = sess.run(self.cos_similarity,
                                feed_dict={self.v: feature_vec, self.vs: self.feature_vectors})
        ixs = np.arange(self.MAX_ELEMENTS_VOLATILE + self.BASE_MEMORY)[:len(self.memory)]
        similarities = similarities[:len(self.memory)]
        similarities = similarities ** alpha
        similarities /= np.sum(similarities)
        ixs = np.random.choice(ixs, batch_size, p=similarities)
        return [self.memory[i] for i in ixs]

    def normalize(self, feature_vec):
        self.feature_vectors_max = np.maximum(self.feature_vectors_max, feature_vec)
        self.feature_vectors_min = np.minimum(self.feature_vectors_min, feature_vec)
        self.feature_vectors_mean += 1 / (self.feature_vectors.shape[0] / 10) * \
                                     (feature_vec - self.feature_vectors_mean)
        feature_vec = (feature_vec - self.feature_vectors_mean) / \
                               (self.feature_vectors_max - self.feature_vectors_min)
        return feature_vec

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


class FrameProcessor:
    def __init__(self, input_shape):
        i = tf.keras.layers.Input(input_shape, dtype=tf.uint8, batch_size=1)
        x = tf.keras.applications.vgg16.preprocess_input(tf.cast(i, tf.float32))
        core = tf.keras.applications.VGG16(include_top=False)
        core = tf.keras.Sequential(core.layers[:7])
        x = core(x)
        out = tf.keras.layers.GlobalAveragePooling2D()(x)
        self.model = tf.keras.Model(inputs=[i], outputs=[out])
        self.model.trainable = False

    def process(self, frame):
        return self.model.predict(np.expand_dims(frame, axis=0))[0]


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
        # Our input are 4 grayscale frames of shape 120, 120 each
        self.X_pl = tf.placeholder(shape=[None, 120, 120, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.lr = tf.placeholder(tf.float32, shape=[])

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.keras.layers.Conv2D(32, 8, 4, activation=tf.nn.relu, kernel_initializer='glorot_normal')(X)
        conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu, kernel_initializer='glorot_normal')(conv1)
        conv3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu, kernel_initializer='glorot_normal')(conv2)

        # Fully connected layers
        flattened = tf.keras.layers.Flatten()(conv3)
        fc1 = tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer='glorot_normal')(flattened)
        self.predictions = tf.keras.layers.Dense(len(VALID_ACTIONS), activation=None,
                                                 kernel_initializer='glorot_normal')(fc1)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(self.lr, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # Summaries for Tensorboard

    #         self.summaries = tf.summary.merge([
    #             tf.summary.scalar("loss", self.loss),
    #             tf.summary.histogram("loss_hist", self.losses),
    #             tf.summary.histogram("q_values_hist", self.predictions),
    #             tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
    #         ])

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
        return sess.run(self.predictions, {self.X_pl: s})

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
        global_step,  _, loss = sess.run(
            [tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict)
        return loss


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

    def policy_fn(sess, observation, epsilon, batch_size):
        if batch_size:
            A = np.ones((batch_size, nA), dtype=float) * epsilon / nA
            q_values = estimator.predict(sess, observation)
            best_actions = np.argmax(q_values, axis=1)
            A[np.arange(batch_size), best_actions] += (1.0 - epsilon)
        else:
            A = np.ones(nA, dtype=float) * epsilon / nA
            q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
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

    processor = FrameProcessor((120, 120, 3))
    replay_memory = RelatedMemoryReplay(128, replay_memory_size)
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
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
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    min_loss = 1e-5
    print("Training started!")
    for i_episode in range(num_episodes):
        ep_reward = 0
        # Save the current checkpoint
        if (i_episode + 1) % 25 == 0:
            saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment
        elapsed_time = time.perf_counter()
        env.retry()
        state, _, _, fr = env.step(0)
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.Canny(state, threshold1=275, threshold2=300)
        state = np.stack([state] * 4, axis=2)
        framerate = 0
        frames_to_write = [env.record_frame]

        loss = 0
        update_num = 0.001
        # One step in the environment
        for t in count(start=1):

            # Epsilon for this time step
            epsilon = 0

            # Maybe update the target estimator
            if (total_t + 1) % update_target_estimator_every == 0:
                copier.make(sess)

            # Take a step in the environment
            action_probs, action_values = policy(sess, state, epsilon, None)
            action = np.random.choice(VALID_ACTIONS, p=action_probs)
            next_frame, reward, done, fr = env.step(action)
            framerate += 1
            if (i_episode + 1) % 100 == 0:
                frames_to_write.append(env.record_frame)
            print(f"\rAction values: {action_values}, total_t: {total_t}", end="")

            # Save transition to replay memory
            next_frame_g = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
            image = cv2.Canny(next_frame_g, threshold1=275, threshold2=300)
            next_state = np.append(state[:, :, 1:], np.expand_dims(image, axis=2), axis=2)
            feature_vec = processor.process(next_frame)
            feature_vec = replay_memory.normalize(feature_vec)
            replay_memory.add(feature_vec, Transition(state, action, reward, next_state, done))
            ep_reward += reward

            # Sample a minibatch from the replay memory
            samples = replay_memory.sample_like(feature_vec, batch_size, sess, alpha=0.7)
            # samples, sample_vectors = replay_memory.sample_random(batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # This is where Double Q-Learning comes in!
            q_values_next = q_estimator.predict(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            td_targets = reward_batch + (1 - done_batch) * \
                         discount_factor * q_values_next_target[np.arange(len(best_actions)), best_actions]

            # # Estimate loss
            batch_loss = q_estimator.calc_loss(sess, states_batch, action_batch, td_targets)
            if batch_loss > min_loss:
                # Perform gradient descent update
                loss += q_estimator.update(sess, states_batch, action_batch, td_targets, 0.00025)
                update_num += 1
                total_t += 1

            if done:
                print("\n" + "-" * 50)
                break
            else:
                state = next_state
        loss /= update_num
        elapsed_time = time.perf_counter() - elapsed_time
        framerate /= elapsed_time
        print(f"\nEpisode's framerate: {framerate}")
        print(f"Replay memory len : {len(replay_memory)}")
        print(f"Current minimal loss: {min_loss}")

        # Wriring episode to video:
        if (i_episode + 1) % 100 == 0:
            height, width, _ = frames_to_write[0].shape
            output = cv2.VideoWriter(os.path.join(videos_dir, f'episode{i_episode + 1}.avi'),
                                     cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
            for frame in frames_to_write:
                output.write(frame.astype('uint8'))
            output.release()

        yield total_t, loss

    return 0


env = gd_utils.GDenv((800, 600))
VALID_ACTIONS = [0, 1]
tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/Geometry Dash")
videos_dir = os.path.abspath("./videos")

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# Run it!
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    plt.ion()
    fig, axs = plt.subplots()
    manager = plt.get_current_fig_manager()
    geom = manager.window.geometry()
    _, _, dx, dy = geom.getRect()
    manager.window.setGeometry(900, 0, dx, dy)
    # axs.set_ylim([0, 40])
    for i, (t, ep_reward) in enumerate(deep_q_learning(sess,
                                                       env,
                                                       q_estimator=q_estimator,
                                                       target_estimator=target_estimator,
                                                       experiment_dir=experiment_dir,
                                                       num_episodes=10_000,
                                                       replay_memory_size=90_000,
                                                       replay_memory_init_size=5_000,
                                                       update_target_estimator_every=10_000,
                                                       discount_factor=0.9,
                                                       batch_size=32)
                                       ):
        print(f"Episode's #{i + 1} loss: {ep_reward}")
        axs.plot(i, ep_reward, "ro")
        fig.canvas.draw()
        fig.canvas.flush_events()

print("Training done!")
plt.savefig("loss_func.png")

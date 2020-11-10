from itertools import count
import numpy as np
import os
import tensorflow as tf
import gd_utils
from random import sample
import time
import cv2
from collections import deque, namedtuple

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


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
        return A

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
                    epsilon_decay_steps=3_000_000,
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

    # The replay memory
    replay_memory_main = dict()
    # replay_memory_dynamic = deque()
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

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    env.retry()
    state, _, _, fr = env.step(0)
    state = np.stack([state] * 4, axis=2)
    a_vec = [0, 0, 0, 0]
    for i in range(replay_memory_init_size):
        # Populate replay memory!
        action_probs = policy(sess, state, epsilons[min(epsilon_decay_steps - 1, total_t)], None)
        action = np.random.choice(VALID_ACTIONS, p=action_probs)
        next_frame, reward, done, fr = env.step(action)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_frame, 2), axis=2)
        put_to_mem(replay_memory_main, fr, a_vec, action, Transition(state, action, reward, next_state, done))
        # replay_memory_dynamic.append(Transition(state, action, reward, next_state, done))
        a_vec = a_vec[1:] + [action]

        if done:
            env.retry()
            state, _, _, fr = env.step(0)
            state = np.stack([state] * 4, axis=2)
            a_vec = [0, 0, 0, 0]
        else:
            state = next_state

    # print("Loading pre-created memory samples...")
    # with open("dynamic.pickle", "rb") as f:
    #     replay_memory_dynamic = deque(load(f))
    # with open("static.pickle", "rb") as f:
    #     replay_memory_main = load(f)
    # print(f"Dynamic: {len(replay_memory_dynamic)}, Static: {len(replay_memory_main)}")
    # Record videos
    # LATER

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
        state = np.stack([state] * 4, axis=2)
        a_vec = [0, 0, 0, 0]
        framerate = 0
        frames_to_write = [env.record_frame]

        # One step in the environment
        for t in count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            # epsilon = 0.03

            # Maybe update the target estimator
            if (total_t + 1) % update_target_estimator_every == 0:
                copier.make(sess)

            # Take a step in the environment
            action_probs = policy(sess, state, epsilon, None)
            action_values = q_estimator.predict(sess, np.expand_dims(state, 0))[0]
            action = np.random.choice(VALID_ACTIONS, p=action_probs)
            next_frame, reward, done, fr = env.step(action)
            framerate += 1
            if (i_episode + 1) % 100 == 0:
                frames_to_write.append(env.record_frame)
            print(f"\rAction values: {action_values}, total_t: {total_t}", end="")

            # If our replay memory is full, pop the first element
            # if len(replay_memory_dynamic) > 1_000:
            #     replay_memory_dynamic.popleft()

            # Save transition to replay memory
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_frame, axis=2), axis=2)
            put_to_mem(replay_memory_main, fr, a_vec, action, Transition(state, action, reward, next_state, done))
            # replay_memory_dynamic.append(Transition(state, action, reward, next_state, done))
            a_vec = a_vec[1:] + [action]
            ep_reward += reward

            # Sample a minibatch from the replay memory
            samples = sample(list(replay_memory_main.values()), batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets
            # q_values_next = target_estimator.predict(sess, next_states_batch)
            # td_targets = reward_batch + discount_factor * np.invert(done_batch).astype(np.float32) * q_values_next.max(
            #     axis=1)

            # This is where Double Q-Learning comes in!
            q_values_next = q_estimator.predict(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            td_targets = reward_batch + (1 - done_batch) * \
                         discount_factor * q_values_next_target[np.arange(len(best_actions)), best_actions]

            # Perform SARSA update:
            # q_values_next = target_estimator.predict(sess, next_states_batch)
            # actions_next_p = policy(sess, states_batch, epsilon, batch_size)
            # actions_next = [np.random.choice(VALID_ACTIONS, p=probs) for probs in actions_next_p]
            # td_targets = reward_batch + discount_factor * (1 - done_batch)\
            #              * q_values_next[np.arange(batch_size), actions_next]

            # TODO Perform gradient descent update
            loss = q_estimator.update(sess, states_batch, action_batch, td_targets, 1 / len(replay_memory_main))
            # print(f"\rTotal t: {total_t}, loss: {loss}", end="")

            if done:
                print("\n" + "-" * 50)
                break
            else:
                state = next_state
                total_t += 1

        elapsed_time = time.perf_counter() - elapsed_time
        framerate /= elapsed_time
        print(f"\nEpisode's framerate: {framerate}")
        # print(f"Replay memory dynamic len : {len(replay_memory_dynamic)}")
        print(f"Replay memory main len : {len(replay_memory_main)}")

        # Wriring episode to video:
        if (i_episode + 1) % 100 == 0:
            height, width, _ = frames_to_write[0].shape
            output = cv2.VideoWriter(os.path.join(videos_dir, f'episode{i_episode + 1}.avi'),
                                     cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
            for frame in frames_to_write:
                output.write(frame)
            output.release()

        # if (i_episode + 1) % 25 == 0:
        #     print("Updating weights (main)...")
        #     total_loss = 0.
        #     for e in range(10):
        #         print(f"\nEpoch #{e}:")
        #         random_batches = random_mini_batches(replay_memory_dynamic, batch_size)
        #         for batch in random_batches:
        #             states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*batch))
        #             q_values_next = target_estimator.predict(sess, next_states_batch)
        #             td_targets = reward_batch + discount_factor * np.invert(done_batch).astype(
        #                 np.float32) * q_values_next.max(axis=1)
        #             loss, grads = q_estimator.update(sess, states_batch, action_batch, td_targets)
        #             print(f"\rBatch loss: {loss}", end="")
        #             total_loss += loss / len(random_batches)
        #     copier.make(sess)
        #     replay_memory_dynamic.clear()
        #     print(f"\nTotal loss: {total_loss}")

        yield total_t, ep_reward

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
    for i, (t, ep_reward) in enumerate(deep_q_learning(sess,
                                                       env,
                                                       q_estimator=q_estimator,
                                                       target_estimator=target_estimator,
                                                       experiment_dir=experiment_dir,
                                                       num_episodes=10_000,
                                                       replay_memory_size=125_000,
                                                       replay_memory_init_size=1_000,
                                                       update_target_estimator_every=10_000,
                                                       discount_factor=0.99,
                                                       batch_size=32)
                                       ):
        print(f"Episode's #{i + 1} reward: {ep_reward}")
    print("Training done!")

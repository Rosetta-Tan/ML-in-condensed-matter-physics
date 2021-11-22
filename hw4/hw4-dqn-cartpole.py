"""
We thank Huskarl and OpenAI gym for paving the basis of this example code.

Reinforcement learning for CartPole problem.
Please run the program, tune parameters and plot the figures of episode rewards during training and testing respectively.
For the parameter 'instance', we recommend setting it to 1, because when it is greater than 1, the convergence will become difficult.
Submit the runtime outputs and the best figures on the course.pku.edu.cn.
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import numpy as np
import random
from collections import namedtuple
import matplotlib.pyplot as plt

import gym

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])
EPS0 = 1e-3 # Constant added to all priorities to prevent them from being zero


class Agent:
	"""Abstract base class for all implemented agents.

	Do not use this abstract base class directly but instead use one of the concrete agents implemented.

	To implement your own agent, you have to implement the following methods:
	"""
	def save(self, filename, overwrite=False):
		"""Saves the model parameters to the specified file."""
		raise NotImplementedError()

	def act(self, state, instance=0):
		"""Returns the action to be taken given a state."""
		raise NotImplementedError()

	def push(self, transition, instance=0):
		"""Stores the transition in memory."""
		raise NotImplementedError()

	def train(self, step):
		"""Trains the agent for one step."""
		raise NotImplementedError()


class DQN(Agent):
	"""Deep Q-Learning Network

	Base implementation:
		"Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)

	Extensions:
		Multi-step returns: "Reinforcement Learning: An Introduction" 2nd ed. (Sutton & Barto, 2018)
		Double Q-Learning: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
		Dueling Q-Network: "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
	"""
	def __init__(self, model, actions, optimizer=None, policy=None, test_policy=None,
				 memsize=10_000, target_update=10, gamma=0.99, batch_size=64, nsteps=1,
				 enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg'):
		"""
		TODO: Describe parameters
		"""
		self.actions = actions
		self.optimizer = Adam(lr=3e-3) if optimizer is None else optimizer

		self.policy = EpsGreedy(0.1) if policy is None else policy
		self.test_policy = Greedy() if test_policy is None else test_policy

		self.memsize = memsize
		self.memory = PrioritizedExperienceReplay(memsize, nsteps)

		self.target_update = target_update
		self.gamma = gamma
		self.batch_size = batch_size
		self.nsteps = nsteps
		self.training = True

		# Extension options
		self.enable_double_dqn = enable_double_dqn
		self.enable_dueling_network = enable_dueling_network
		self.dueling_type = dueling_type

		# Create output layer based on number of actions and (optionally) a dueling architecture
		raw_output = model.layers[-1].output
		if self.enable_dueling_network:
			# "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
			# Output the state value (V) and the action-specific advantages (A) separately then compute the Q values: Q = A + V
			dueling_layer = Dense(self.actions + 1, activation='linear')(raw_output)
			if   self.dueling_type == 'avg':   f = lambda a: tf.expand_dims(a[:,0], -1) + a[:,1:] - tf.reduce_mean(a[:,1:], axis=1, keepdims=True)
			elif self.dueling_type == 'max':   f = lambda a: tf.expand_dims(a[:,0], -1) + a[:,1:] - tf.reduce_max(a[:,1:], axis=1, keepdims=True)
			elif self.dueling_type == 'naive': f = lambda a: tf.expand_dims(a[:,0], -1) + a[:,1:]
			output_layer = Lambda(f, output_shape=(self.actions,))(dueling_layer)
		else:
			output_layer = Dense(self.actions, activation='linear')(raw_output)

		self.model = Model(inputs=model.input, outputs=output_layer)

		# Define loss function that computes the MSE between target Q-values and cumulative discounted rewards
		# If using PrioritizedExperienceReplay, the loss function also computes the TD error and updates the trace priorities
		def masked_q_loss(data, y_pred):
			"""Computes the MSE between the Q-values of the actions that were taken and	the cumulative discounted
			rewards obtained after taking those actions. Updates trace priorities if using PrioritizedExperienceReplay.
			"""
			action_batch, target_qvals = data[:, 0], data[:, 1]
			seq = tf.cast(tf.range(0, tf.shape(action_batch)[0]), tf.int32)
			action_idxs = tf.transpose(tf.stack([seq, tf.cast(action_batch, tf.int32)]))
			qvals = tf.gather_nd(y_pred, action_idxs)
			if isinstance(self.memory, PrioritizedExperienceReplay):
				def update_priorities(_qvals, _target_qvals, _traces_idxs):
					"""Computes the TD error and updates memory priorities."""
					td_error = np.abs((_target_qvals - _qvals).numpy())
					_traces_idxs = (tf.cast(_traces_idxs, tf.int32)).numpy()
					self.memory.update_priorities(_traces_idxs, td_error)
					return _qvals
				qvals = tf.py_function(func=update_priorities, inp=[qvals, target_qvals, data[:,2]], Tout=tf.float32)
			return tf.keras.losses.mse(qvals, target_qvals)

		self.model.compile(optimizer=self.optimizer, loss=masked_q_loss)

		# Clone model to use for delayed Q targets
		self.target_model = tf.keras.models.clone_model(self.model)
		self.target_model.set_weights(self.model.get_weights())

	def save(self, filename, overwrite=False):
		"""Saves the model parameters to the specified file."""
		self.model.save_weights(filename, overwrite=overwrite)

	def act(self, state, instance=0):
		"""Returns the action to be taken given a state."""
		qvals = self.model.predict(np.array([state]))[0]
		return self.policy.act(qvals) if self.training else self.test_policy.act(qvals)

	def push(self, transition, instance=0):
		"""Stores the transition in memory."""
		self.memory.put(transition)

	def train(self, step):
		"""Trains the agent for one step."""
		if len(self.memory) == 0:
			return None

		# Update target network
		if self.target_update >= 1 and step % self.target_update == 0:
			# Perform a hard update
			self.target_model.set_weights(self.model.get_weights())
		elif self.target_update < 1:
			# Perform a soft update
			mw = np.array(self.model.get_weights())
			tmw = np.array(self.target_model.get_weights())
			self.target_model.set_weights(self.target_update * mw + (1 - self.target_update) * tmw)

		# Train even when memory has fewer than the specified batch_size
		batch_size = min(len(self.memory), self.batch_size)

		# Sample batch_size traces from memory
		state_batch, action_batch, reward_batches, end_state_batch, not_done_mask = self.memory.get(batch_size)

		# Compute the value of the last next states
		target_qvals = np.zeros(batch_size)
		non_final_last_next_states = [es for es in end_state_batch if es is not None]

		if len(non_final_last_next_states) > 0:
			if self.enable_double_dqn:
				# "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
				# The online network predicts the actions while the target network is used to estimate the Q-values
				q_values = self.model.predict_on_batch(np.array(non_final_last_next_states))
				actions = np.argmax(q_values, axis=1)
				# Estimate Q-values using the target network but select the values with the
				# highest Q-value wrt to the online model (as computed above).
				target_q_values = self.target_model.predict_on_batch(np.array(non_final_last_next_states))
				selected_target_q_vals = target_q_values[range(len(target_q_values)), actions]
			else:
				# Use delayed target network to compute target Q-values
				selected_target_q_vals = self.target_model.predict_on_batch(np.array(non_final_last_next_states)).max(1)
			non_final_mask = list(map(lambda s: s is not None, end_state_batch))
			target_qvals[non_final_mask] = selected_target_q_vals

		# Compute n-step discounted return
		# If episode ended within any sampled nstep trace - zero out remaining rewards
		for n in reversed(range(self.nsteps)):
			rewards = np.array([b[n] for b in reward_batches])
			target_qvals *= np.array([t[n] for t in not_done_mask])
			target_qvals = rewards + (self.gamma * target_qvals)

		# Compile information needed by the custom loss function
		loss_data = [action_batch, target_qvals]

		# If using PrioritizedExperienceReplay then we need to provide the trace indexes
		# to the loss function as well so we can update the priorities of the traces
		if isinstance(self.memory, PrioritizedExperienceReplay):
			loss_data.append(self.memory.last_traces_idxs())

		# Train model
		self.model.train_on_batch(np.array(state_batch), np.stack(loss_data).transpose())


class Policy:
	"""Abstract base class for all implemented policies.

	Do not use this abstract base class directly but instead use one of the concrete policies implemented.

	A policy ultimately returns the action to be taken based on the output of the agent.
	The policy is the place to implement action-space exploration strategies.
	If the action space is discrete, the policy usually receives action values and has to pick an action/index.
	A discrete action-space policy can explore by pick an action at random with a small probability e.g. EpsilonGreedy.
	If the action space is continuous, the policy usually receives a single action or a distribution over actions.
	A continuous action-space policy can simply sample from the distribution and/or add noise to the received action.

	To implement your own policy, you have to implement the following method:
	"""
	def act(self, **kwargs):
		raise NotImplementedError()


class Greedy(Policy):
	"""Greedy Policy

	This policy always picks the action with largest value.
	"""
	def act(self, qvals):
		return np.argmax(qvals)


class EpsGreedy(Policy):
	"""Epsilon-Greedy Policy

	This policy picks the action with largest value with probability 1-epsilon.
	It picks a random action and therefore explores with probability epsilon.
	"""
	def __init__(self, eps):
		self.eps = eps

	def act(self, qvals):
		if random.random() > self.eps:
			return np.argmax(qvals)
		return random.randrange(len(qvals))


class PrioritizedExperienceReplay:
	"""Stores prioritized interaction with an environment in a priority queue implemented via a heap.

	Provides efficient prioritized sampling of multistep traces.
	If exclude_boundaries==True, then traces are sampled such that they don't include episode boundaries.
	For more information see "Prioritized Experience Replay" (Schaul et al., 2016).
	"""
	def __init__(self, capacity, steps=1, exclude_boundaries=False, prob_alpha=0.6):
		"""
		Args:
			capacity (int): The maximum number of traces the memory should be able to store.
			steps (int): The number of steps (transitions) each sampled trace should include.
			exclude_boundaries (bool): If True, sampled traces will not include episode boundaries.
			prob_alpha (float): Value between 0 and 1 that specifies how strongly priorities are taken into account.
		"""
		self.traces = [] # Each element is a tuple containing self.steps transitions
		self.priorities = np.array([]) # Each element is the priority for the same-index trace in self.traces
		self.buffer = [] # Rolling buffer of size at most self.steps
		self.capacity = capacity
		self.steps = steps
		self.exclude_boundaries = exclude_boundaries
		self.prob_alpha = prob_alpha
		self.traces_idxs = [] # Temporary list that contains the indexes associated to the last retrieved traces

	def put(self, transition):
		"""Adds transition to memory."""
		# Append transition to temporary rolling buffer
		self.buffer.append(transition)
		# If buffer doesn't yet contain a full trace - return
		if len(self.buffer) < self.steps: return
		# If self.traces not at max capacity, append new trace and priority (use highest existing priority if available)
		if len(self.traces) < self.capacity:
			self.traces.append(tuple(self.buffer))
			self.priorities = np.append(self.priorities, EPS0 if self.priorities.size == 0 else self.priorities.max())
		else:
			# If self.traces at max capacity, substitute lowest priority trace and use highest existing priority
			idx = np.argmin(self.priorities)
			self.traces[idx] = tuple(self.buffer)
			self.priorities[idx] = self.priorities.max()
		# If excluding boundaries and we've reached a boundary - clear the buffer
		if self.exclude_boundaries and transition.next_state is None:
			self.buffer = []
			return
		# Roll buffer
		self.buffer = self.buffer[1:]

	def get(self, batch_size):
		"""Samples the specified number of traces from the buffer according to the prioritization and prob_alpha."""
		# Transform priorities into probabilities using self.prob_alpha
		probs = self.priorities ** self.prob_alpha
		probs /= probs.sum()
		# Sample batch_size traces according to probabilities and store indexes
		self.traces_idxs = np.random.choice(len(self.traces), batch_size, p=probs, replace=False)
		traces = [self.traces[idx] for idx in self.traces_idxs]
		return unpack(traces)

	def last_traces_idxs(self):
		"""Returns the indexes associated with the last retrieved traces."""
		return self.traces_idxs.copy()

	def update_priorities(self, traces_idxs, new_priorities):
		"""Updates the priorities of the traces with specified indexes."""
		self.priorities[traces_idxs] = new_priorities + EPS0

	def __len__(self):
		"""Returns the number of traces stored."""
		return len(self.traces)


def unpack(traces):
	"""Returns states, actions, rewards, end_states, and a mask for episode boundaries given traces."""
	states = [t[0].state for t in traces]
	actions = [t[0].action for t in traces]
	rewards = [[e.reward for e in t] for t in traces]
	end_states = [t[-1].next_state for t in traces]
	not_done_mask = [[1 if n.next_state is not None else 0 for n in t] for t in traces]
	return states, actions, rewards, end_states, not_done_mask


class Simulation:
	"""Simulates an agent interacting with one of multiple environments."""
	def __init__(self, create_env, agent, mapping=None):
		self.create_env = create_env
		self.agent = agent
		self.mapping = mapping

	def train(self, max_steps=100_000, instances=1, visualize=False, plot=None):
		"""Trains the agent on the specified number of environment instances."""
		self.agent.training = True
		self._sp_train(max_steps, instances, visualize)

	def _sp_train(self, max_steps, instances, visualize):
		"""Trains using a single process."""
		# Keep track of rewards per episode per instance
		episode_reward_sequences = [[] for _ in range(instances)]
		episode_step_sequences = [[] for _ in range(instances)]
		episode_rewards = [0] * instances

		# Create and initialize environment instances
		envs = [self.create_env() for i in range(instances)]
		states = [env.reset() for env in envs]

		episode = 0
		for step in range(max_steps):
			for i in range(instances):
				if visualize: envs[i].render()
				action = self.agent.act(states[i], i)
				next_state, reward, done, _ = envs[i].step(action)
				self.agent.push(Transition(states[i], action, reward, None if done else next_state), i)
				episode_rewards[i] += reward
				if done:
					print('instances={}\tepisode={}\tstep={}\tepisode reward={}'.format(i, len(
						episode_reward_sequences[i]), step, episode_rewards[i]))
					episode_reward_sequences[i].append(episode_rewards[i])
					episode_step_sequences[i].append(step)
					episode_rewards[i] = 0
					states[i] = envs[i].reset()
				else:
					states[i] = next_state
			# Perform one step of the optimization
			self.agent.train(step)
		for i in range(instances):
			episode_reward_sequences[i].append(episode_rewards[i])
			episode_step_sequences[i].append(max_steps - 1)

		plt.figure()
		color = np.concatenate((np.linspace(1, 0.5, num=instances)[:, np.newaxis],
								np.linspace(0., 0.5, num=instances)[:, np.newaxis], np.zeros((instances, 1))), axis=1)
		for i in range(instances):
			plt.scatter(episode_step_sequences[i], episode_reward_sequences[i],
						c=np.tile(color[i][np.newaxis, :], (len(episode_reward_sequences[i]), 1)))
		plt.xlim(0, max_steps)
		plt.xlabel('Step')
		plt.ylabel('Episode Reward')
		plt.title('Train')
		plt.legend(['instance {}'.format(i) for i in range(instances)])
		plt.savefig("train_fig")
		plt.close()

	def test(self, max_steps, visualize=True):
		"""Test the agent on the environment."""
		self.agent.training = False

		# Create and initialize environment
		env = self.create_env()
		state = env.reset()
		episode_rewards, episode_reward_sequences, episode_step_sequences = 0., [], []
		for step in range(max_steps):
			if visualize: env.render()
			action = self.agent.act(state)
			next_state, reward, done, _ = env.step(action)
			episode_rewards += reward
			if done:
				print('episode={}\tstep={}\tepisode reward={}'.format(len(episode_step_sequences), step, episode_rewards))
				episode_reward_sequences.append(episode_rewards)
				episode_step_sequences.append(step)
				episode_rewards = 0.
				state = env.reset()
			else:
				state = next_state
		episode_reward_sequences.append(episode_rewards)
		episode_step_sequences.append(max_steps - 1)

		plt.figure()
		plt.scatter(episode_step_sequences, episode_reward_sequences)
		plt.xlim(0, max_steps)
		plt.xlabel('Step')
		plt.ylabel('Episode Reward')
		plt.title('Test')
		plt.savefig("test_fig")
		plt.close()


if __name__ == '__main__':

	# Setup gym environment
	create_env = lambda: gym.make('CartPole-v0').unwrapped
	dummy_env = create_env()

	# Build a simple neural network with 3 fully connected layers as our model
	model = Sequential([
		Dense(16, activation='relu', input_shape=dummy_env.observation_space.shape),
		Dense(16, activation='relu'),
		Dense(16, activation='relu'),
	])

	# Create Q-Learning ANN agent
	agent = DQN(model, actions=dummy_env.action_space.n, nsteps=2)

	# Create simulation, train and then test
	sim = Simulation(create_env, agent)
	print('Train')
	sim.train(max_steps=3000, instances=1, visualize=False)
	print('Test')
	sim.test(max_steps=1000, visualize=False)

import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, N, Nt, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.N = N
		self.Nt = Nt

		self.state = np.zeros((max_size, 2*N, 4*Nt+1))
		self.action = np.zeros((max_size, Nt, N+1, 2))
		self.next_state = np.zeros((max_size, 2*N, 4*Nt+1))
		self.reward = np.zeros((max_size, 1))
		self.episode_c= np.zeros((max_size, 1))
		self.episode_start = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, episode_c, episode_start, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.episode_c[self.ptr] = episode_c
		self.episode_start[self.ptr] = episode_start
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.episode_c[ind]).to(self.device),
			torch.FloatTensor(self.episode_start[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def multi_step_sample(self, batch_size, n_step):
		ind = np.random.choice(self.total_ind, size=batch_size, replace = False)

		replay_seq = []
		for step in range(n_step):
			step_ind = ind + step
			step_replay = (
				torch.FloatTensor(self.state[step_ind]).to(self.device),
				torch.FloatTensor(self.action[step_ind]).to(self.device),
				torch.FloatTensor(self.next_state[step_ind]).to(self.device),
				torch.FloatTensor(self.reward[step_ind]).to(self.device),
				torch.FloatTensor(self.episode_c[step_ind]).to(self.device),
				torch.FloatTensor(self.episode_start[step_ind]).to(self.device),
				torch.FloatTensor(self.not_done[step_ind]).to(self.device)
			)
			replay_seq.append(step_replay)
		
		return replay_seq

	def save(self):
		np.save('replay_state.npy', self.state[:self.ptr])
		np.save('replay_action.npy', self.action[:self.ptr])
		np.save('replay_next_state.npy', self.next_state[:self.ptr])
		np.save('replay_reward.npy', self.reward[:self.ptr])
		np.save('replay_episode_c.npy', self.episode_c[:self.ptr])
		np.save('replay_episode_start.npy', self.episode_start[:self.ptr])
		np.save('replay_not_done.npy', self.not_done[:self.ptr])
	
	def load(self, n_step):
		self.state = np.load('replay_state.npy')
		self.action = np.load('replay_action.npy')
		self.next_state = np.load('replay_next_state.npy')
		self.reward = np.load('replay_reward.npy')
		self.episode_c= np.load('replay_episode_c.npy')
		self.episode_start = np.load('replay_episode_start.npy')
		self.not_done = np.load('replay_not_done.npy')
		current_len = self.episode_c.shape[0]
		self.size = current_len
		self.ptr = current_len

		episode_start_indx = np.where(self.episode_start[:,0]==1)[0]
		self.total_ind = set(np.arange(self.size-40))
		for step in range(n_step+1):
			self.total_ind = self.total_ind - set(episode_start_indx -step - 1)
		
		self.total_ind = list(self.total_ind)

	def update_total_ind(self,n_step):
		episode_start_indx = np.where(self.episode_start[:,0]==1)[0]
		self.total_ind = set(np.arange(self.size-40))
		for step in range(n_step+1):
			self.total_ind = self.total_ind - set(episode_start_indx -step - 1)
		
		self.total_ind = list(self.total_ind)		

	def save_explore(self):
		np.save('explore_state.npy', self.state[:self.ptr])
		np.save('explore_action.npy', self.action[:self.ptr])
		np.save('explore_next_state.npy', self.next_state[:self.ptr])
		np.save('explore_reward.npy', self.reward[:self.ptr])
		np.save('explore_episode_c.npy', self.episode_c[:self.ptr])
		np.save('explore_episode_start.npy', self.episode_start[:self.ptr])
		np.save('explore_not_done.npy', self.not_done[:self.ptr])

	def load_explore(self, n_step):
		self.state = np.load('explore_state.npy')
		self.action = np.load('explore_action.npy')
		self.next_state = np.load('explore_next_state.npy')
		self.reward = np.load('explore_reward.npy')
		self.episode_c= np.load('explore_episode_c.npy')
		self.episode_start = np.load('explore_episode_start.npy')
		self.not_done = np.load('explore_not_done.npy')
		current_len = self.state.shape[0]
		self.size = current_len
		self.ptr = current_len

		episode_start_indx = np.where(self.episode_start[:,0]==1)[0]
		self.total_ind = set(np.arange(self.size-40))
		for step in range(n_step+1):
			self.total_ind = self.total_ind - set(episode_start_indx -step - 1)
		
		self.total_ind = list(self.total_ind)

		# Create empty replay for storing additional transition
		padding_size = self.max_size - self.size
		N = self.N
		Nt = self.Nt

		state_ = np.zeros((padding_size, 2*N, 4*Nt+1))
		action_ = np.zeros((padding_size, Nt, N+1, 2))
		next_state_ = np.zeros((padding_size, 2*N, 4*Nt+1))
		reward_ = np.zeros((padding_size, 1))
		episode_c_= np.zeros((padding_size, 1))
		episode_start_ = np.zeros((padding_size, 1))
		not_done_ = np.zeros((padding_size, 1))

		self.state = np.concatenate((self.state, state_), axis=0)
		self.action = np.concatenate((self.action, action_), axis=0)
		self.next_state = np.concatenate((self.next_state, next_state_), axis=0)
		self.reward = np.concatenate((self.reward, reward_), axis=0)
		self.episode_c= np.concatenate((self.episode_c, episode_c_), axis=0)
		self.episode_start = np.concatenate((self.episode_start, episode_start_), axis=0)
		self.not_done = np.concatenate((self.not_done, not_done_), axis=0)
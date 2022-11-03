import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from networks import ActorNetwork, CriticNetwork, CriticNetwork_MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mu = 0.00
Nr = 6
Nt = Nr
N = 4
# QoS = 0
# noise_p = 0.001
# lagarange_ini = 0
eval_channel_index = int(2*Nr*N)+int(2*Nr*N)+N

class LinearSchedule:
    def __init__(self, start, end=None, steps=None, start_step=1):
        
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max
        self.total_steps = 0
        self.start_step = start_step

    def __call__(self, steps=1):
        
        val = self.current
        self.total_steps += 1
        if self.total_steps >= self.start_step:
            self.current = self.bound(self.current + self.inc * steps, self.end)
    
        return val

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, action_scale):
        super(Actor, self).__init__()
        hidden_dim = 1024
        embedding_dim = hidden_dim
        n_layers = 2
        # encoder_normalization = 'batch'
        

        hidden_dim_1 = 1024
        hidden_dim_2 = 1024
        # res_dim_1 = 200
        # res_dim_2 = 200

        self.l1 = nn.Linear(state_dim, hidden_dim_1)
        print('Action state dim', state_dim)
        self.l2 = nn.Linear(hidden_dim_1, hidden_dim_2)

        # self.res_l1 = nn.Linear(hidden_dim_2, res_dim_1)
        # self.res_l2 = nn.Linear(res_dim_1, res_dim_2)

        # self.res_l3 = nn.Linear(res_dim_2, res_dim_2)
        # self.res_l4 = nn.Linear(res_dim_2, res_dim_2)

        self.l3 = nn.Linear(hidden_dim_2, action_dim)
        # self.QoSlayer = QoSLayer(N, Nr)
        self.action_scale = action_scale
        self.max_action = max_action
        # self.error_tole = 1e-3
        
        self.cnn_actor = ActorNetwork(state_dim, action_dim, \
            embedding_dim, hidden_dim, n_layers, Nt, N).to(device)

    def forward(self, state):
        # Vanilla TD3 Actor 
        # print('actor forward', state, state.shape)
        batch_size = state.size(0)
        # a = F.relu(self.l1(state))
        a = self.cnn_actor(state)
        a = self.max_action * torch.tanh(a)
        power = torch.sum(torch.pow(a,2), dim=(1,2,3)).reshape(batch_size, 1, 1, 1)
        scale = torch.sqrt(1/power)
        a = a * scale
        # power_new = torch.sum(torch.pow(a,2), dim=(1,2,3)).reshape(batch_size, 1, 1, 1)

        return a.float()


class Critic(nn.Module): # Critic networks
    def __init__(self, state_dim, action_dim, action_scale):
        super(Critic, self).__init__()

        self.action_scale = action_scale
        hidden_dim = 1024
        embedding_dim = hidden_dim
        n_layers = 2

        # V = 6 critic networks to alleviate overestimation

        self.cnn_critic1 = CriticNetwork(state_dim, action_dim, \
            embedding_dim, hidden_dim, n_layers, Nt, N).to(device)

        self.cnn_critic2 = CriticNetwork(state_dim, action_dim, \
            embedding_dim, hidden_dim, n_layers, Nt, N).to(device)

        self.cnn_critic3 = CriticNetwork(state_dim, action_dim, \
            embedding_dim, hidden_dim, n_layers, Nt, N).to(device)

        self.cnn_critic4 = CriticNetwork(state_dim, action_dim, \
            embedding_dim, hidden_dim, n_layers, Nt, N).to(device)

        self.cnn_critic5 = CriticNetwork(state_dim, action_dim, \
            embedding_dim, hidden_dim, n_layers, Nt, N).to(device)
    
        self.cnn_critic6 = CriticNetwork(state_dim, action_dim, \
            embedding_dim, hidden_dim, n_layers, Nt, N).to(device)

    def forward(self, state, action): # forward prop.

        input_state = self.construct_(state,action) # CNN Critic

        q1 = self.cnn_critic1(input_state)
        q2 = self.cnn_critic2(input_state)
        q3 = self.cnn_critic3(input_state)
        q4 = self.cnn_critic4(input_state)
        q5 = self.cnn_critic5(input_state)
        q6 = self.cnn_critic6(input_state)

        return q1, q2, q3, q4, q5, q6


    def Q1(self, state, action):

        # batch_size = state.size(0)                  # MLP Critic
        # state = state.flatten(start_dim=1)
        # action = action.flatten(start_dim=1)
        # input_state = torch.cat([state, action], 1)

        input_state = self.construct_(state,action)     #CNN Critic
        return self.cnn_critic1(input_state)

    def construct_(self, state, action): # Construct the input for the critic networks based on the state and action tensors

        batch_size = state.size(0)
        node_size = state.size(1)
        CSI_size = state.size(2)
        H = state[:,:,:int(2*Nt)]
        ori_state = state.view(batch_size, node_size, 4, int(CSI_size/4))
        ori_state = torch.permute(ori_state, (0,2,1,3))

        action = torch.transpose(action, 1, 2)
        common_bfv = action[:,0:1,:,:]
        private_bfv = action[:,1:,:,:]
        common_bfv_flat = torch.cat((common_bfv[:,:,:,0], common_bfv[:,:,:,1]), dim=2)        
        common_bfv_new = common_bfv_flat.repeat(1,N,1)
        private_bfv_new = torch.cat((private_bfv[:,:,:,0], private_bfv[:,:,:,1]), dim=2)
        bfv_new = torch.cat((common_bfv_new, private_bfv_new), dim=1).to(device) 

        next_bfv = bfv_new
        new_state = torch.cat((H, next_bfv), dim=2)
        new_state = new_state.view(batch_size, node_size, 4, int(CSI_size/4))
        new_state = torch.permute(new_state, (0,2,1,3))

        return new_state
        


class Agent(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.001,
        policy_noise=0.02,
        noise_clip=0.02,
        policy_freq=2,
        action_scale = 0.003,
        w1 = 1e-3,
        w2 = 1
    ):
        self.eps_start_step = 3e6
        self.actor = Actor(state_dim, action_dim, max_action, action_scale).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-4, weight_decay=1e-6)

        self.robust_eps_logger = 0.
        self.robust_loss_logger = 0.
        self.target_Q_value_logger = 0.
        self.actor_loss_logger = 0.
        self.eps_steps = 1.0
        self.kappa = 1.

        self.w1 = w1
        self.w2 = w2

        self.critic = Critic(state_dim, action_dim, action_scale).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-6)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

        self.action_scale = action_scale

        self.crt = nn.MSELoss()
        self.n_step = 5 # n-step bootstrapping

    def select_action(self, state): # Retrieve action vector based on the forward prop. of actor network
        with torch.no_grad():       # No need to retain gradient here
            state = torch.FloatTensor(state).to(device)
            state = state[:,:,:int(4*Nt)]
            batch_size = state.size(0)
            node_size = state.size(1)
            CSI_size = state.size(2)
            state_attn_cnn = state.view(batch_size,node_size,4, int(CSI_size/4))
            state_attn_cnn = torch.permute(state_attn_cnn, (0,2,1,3))

            true_return = self.actor(state_attn_cnn)
            true_return = true_return[0,:,:,:]
            true_return = true_return.cpu().data.numpy()

        self.actor.eval()
        return true_return

    def train(self, agent_replay, expert_replay, batch_size): # Training algorithm
        self.total_it += 1
        self.actor.train()

        # Sample experience replay
        replay_seq = agent_replay.multi_step_sample(batch_size, self.n_step)  

        with torch.no_grad():
            # Compute the n-step return 
            n_step_reward = torch.zeros(batch_size, device=device, dtype=torch.float).unsqueeze(dim=1)
            for step in range(self.n_step-1):
                
                _, _, _, reward_0, _, _, _ = replay_seq[step]
                n_step_reward = n_step_reward + self.discount**(step) * reward_0
                    
            # The last step of n-step
            state_1, action_1, next_state_1, _, _, _, not_done_1 = replay_seq[self.n_step-1]
            
            # Obtain the input for the critic network from the input tensors
            critic_next_state = state_1[:,:,:int(4*Nt)]
            next_state_1 = self.reshape_state(next_state_1)

            # Add noise to avoid overestimation, see Policy Smoothing in https://arxiv.org/abs/1802.09477
            noise = (
                torch.randn_like(action_1) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)   

            # Determine the next action based on the actor network's policy
            state_1 = self.reshape_state(state_1)    
            next_action = (
                self.actor_target(state_1) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q-value
            target_Q1, target_Q2, target_Q3, target_Q4, target_Q5, target_Q6 \
                = self.critic_target(critic_next_state, next_action)
            target_Q_cat = torch.cat((target_Q1, target_Q2, target_Q3, target_Q4, target_Q5, target_Q6), dim=1)
            target_Q = torch.amin(target_Q_cat, dim=1, keepdim=True)
            target_Q = n_step_reward + not_done_1 * self.discount**(self.n_step) * target_Q
            self.target_Q_value_logger = target_Q.mean().item()   

        # Get the current Q-value estimates from the V critic networks
        state_cr, action_cr, _, _, _, _, _ = replay_seq[0] # current state
        critic_state_cr = state_cr[:,:,:int(4*Nt)]
        current_Q1, current_Q2, current_Q3, current_Q4, current_Q5, current_Q6 \
            = self.critic(critic_state_cr, action_cr)

        # Compute the loss for updating critic networks
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) \
            + F.mse_loss(current_Q3, target_Q) + F.mse_loss(current_Q5, target_Q) \
            + F.mse_loss(current_Q4, target_Q) + F.mse_loss(current_Q6, target_Q)

        # Optimize the critic networks
        self.robust_loss_logger = critic_loss.item()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()	

        # Delayed actor network updates
        if self.total_it % self.policy_freq == 0:

            # Sample from the demonstration replay for imitation loss
            state_exp, action_exp, _, _, _, _, _ = expert_replay.sample(batch_size)
            # Prepare the input 
            state_exp = self.reshape_state(state_exp)
            action_on_expert_replay = self.actor(state_exp)

            # Compute the imitation loss
            imitation_loss = self.crt(action_on_expert_replay, action_exp)

            # Compute the deterministic policy gradient (DPG)
            state = self.reshape_state(state_cr)
            actor_action = self.actor(state)
            actor_loss_ori = -self.critic.Q1(critic_state_cr, actor_action).mean()

            # Combined two losses
            actor_loss = self.w1*actor_loss_ori + self.w2*imitation_loss 
            self.actor_loss_logger = actor_loss.item() # Some logging funcs (optional)

            # Optimize the actor 
            self.actor.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return self.robust_loss_logger, self.actor_loss_logger, self.target_Q_value_logger


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def reshape_state(self, state): # Reshape the input tensors before feeding into the networks
            state = state[:,:,:int(4*Nt)]
            batch_size = state.size(0)
            node_size = state.size(1)
            CSI_size = state.size(2)
            state = state.view(batch_size,node_size, 4, int(CSI_size/4))
            state = torch.permute(state, (0,2,1,3))  
            return state


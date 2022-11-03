import numpy as np
import torch
import argparse
import os
import math
import logging
import copy
from utils import ReplayBuffer

import matlab.engine
eng = matlab.engine.start_matlab()

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_printoptions(precision=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Nt = 6 # Number of Tx Attn.
N = 4 # Number of users
Nr = 1 # Number of Rx Attn.


# Physical env
height = 3.5 # Height of the indoor facility
freq = 60 * (10**9)
sol = 299792458
pathloss_op = 2.29
bandwidth = 10**9
noise_p = 3.9810717055e-21 * bandwidth
normalizer = 1/noise_p
block_length = 256


class Env:

    def __init__(self):
        self.reset()
        
    def reset(self):
        self.demo_freq = 1
        self.env_step_counter = 0
        self.max_step = 40
        self.done_flag = False
        self.current_bfv = np.zeros((Nt,N+1),dtype=np.complexfloating)

        x_cor = np.random.uniform(1, 4, N)
        y_cor = np.random.uniform(1, 4, N)
        hori_dis = np.sqrt(np.power(x_cor,2) + np.power(y_cor,2))
        distance_ = np.sqrt(np.power(hori_dis,2) + np.power(height,2))
        tangent_AOD = np.asmatrix(np.arctan(height/hori_dis)).T

        # Generating AoAs and AoDs
        AoD_user = tangent_AOD
        AoA_user = np.asmatrix(np.random.uniform(-math.pi/4, math.pi/4, N)).T

        # Generate steering vectors
        rx_sv_expand =  np.asmatrix(np.arange(0, Nr)).T
        rx_sv_angle = rx_sv_expand @ np.sin(AoA_user).T
        rx_sv = np.exp(-1j*rx_sv_angle)

        tx_sv_expand =  np.asmatrix(np.arange(0, Nt)).T
        tx_sv_angle = tx_sv_expand @ np.sin(AoD_user).T
        tx_sv = np.exp(-1j*tx_sv_angle)

        complex_gain = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(N, 2)).view(np.complex128)

        distance = np.asmatrix(distance_).T
        pathloss_ = np.sqrt(np.power((4*math.pi*distance*freq)/sol, -pathloss_op)) * np.sqrt(normalizer)

        H = np.zeros((N, Nt), dtype=np.complex128)
        for n in range(N):
            scale_n = (pathloss_[n,:] * complex_gain[n,:])
            H_ = np.multiply(scale_n, (rx_sv[:,n] @ tx_sv[:,n].H))
            H[n,:] = H_

        self.user_channel = H
        H = copy.deepcopy(self.user_channel)

        ini_state_real = np.real(H.flatten())
        ini_state_imag = np.imag(H.flatten())
        ini_state = np.concatenate((ini_state_real, ini_state_imag),0)

        action = np.random.uniform(-1,1,size=(Nt,N+1,2))
        action_power = (np.linalg.norm(action))**2
        tx_bfv_scale = np.sqrt(1/action_power)
        action = tx_bfv_scale * action

        new_state, _, _, _, _, _ = self.step(ini_state, action)
        return new_state, action, False
    
    def action_space(self):
        return Nr
    def step(self, state, action):
        demo_flag = 0
        env_step_counter = self.env_step_counter
        max_step = self.max_step
        current_bfv = copy.deepcopy(self.current_bfv)
        action_delta = action[:,:,0] + 1j*action[:,:,1]
        current_bfv = action_delta
        tx_bfv_power = (np.linalg.norm(current_bfv))**2
        if tx_bfv_power > 0:
            tx_bfv_scale = np.sqrt(1/tx_bfv_power)
            current_bfv = tx_bfv_scale * current_bfv
        tx_bfv_power = (np.linalg.norm(current_bfv))**2
        # print('tx_bfv_power', tx_bfv_power)
        
        current_bfv_real = np.clip(np.real(current_bfv), -1, 1)
        current_bfv_imag = np.clip(np.imag(current_bfv), -1, 1)

        current_bfv = current_bfv_real + 1j*current_bfv_imag
       
        ini_bf = current_bfv.T
        ini_bf = ini_bf[...,np.newaxis]
        # print('current_bfv', current_bfv.shape)

        H = copy.deepcopy(self.user_channel)
        if env_step_counter != 0:
            demo_flag = 1
            real_joint_h = np.real(H)
            imag_joint_h = np.imag(H)
            real_matlab_joint = matlab.double(real_joint_h.tolist())
            imag_matlab_joint = matlab.double(imag_joint_h.tolist())

            ini_bf_np = current_bfv
            ini_bf_real = np.real(ini_bf_np)
            ini_bf_imag = np.imag(ini_bf_np)
            ini_bf_real_matlab = matlab.double(ini_bf_real.tolist())
            ini_bf_imag_matlab = matlab.double(ini_bf_imag.tolist())
            N_matlab = matlab.double([int(N)])

            # Invoke the conventional optimization algorithm
            FP_bfv_real, FP_bfv_imag, solved_flag = eng.opt_algo(real_matlab_joint, imag_matlab_joint, \
                ini_bf_real_matlab, ini_bf_imag_matlab, N_matlab, nargout=3)

            # Receive and construct the solution from the MATLAB Core
            FP_solution = np.array(FP_bfv_real) + 1j*np.array(FP_bfv_imag)
            current_bfv = FP_solution
            self.current_bfv = FP_solution
            
            new_P = FP_solution.T
            tx_bfv = new_P[...,np.newaxis]
            tx_bfv_power = (np.linalg.norm(tx_bfv))**2

            demo_action = np.concatenate((np.array(FP_bfv_real)[...,np.newaxis], np.array(FP_bfv_imag)[...,np.newaxis]),axis=2)
            demo_action_delta = demo_action

            cm_pow = np.zeros(N)
            tar_pow = np.zeros(N)
            itf_pow = np.zeros(N)
            common_rate_vec = np.zeros(N)
            private_rate_vec = np.zeros(N)
            for n in range(N):
                H_n = H[[n],:]
                cm_bfv_ = np.asmatrix(tx_bfv[0,:])
                tx_bfv_ = np.asmatrix(tx_bfv[n+1,:])
                cm_pow[n] = np.square(np.absolute(H_n @ cm_bfv_))
                tar_pow[n] = np.square(np.absolute(H_n @ tx_bfv_))
                itf_pow_temp = 0.

                for j in range(N):
                    if j != n:
                        itf_tx_bfv = np.asmatrix(tx_bfv[j+1,:])
                        itf_pow_temp += np.square(np.absolute(H_n @ itf_tx_bfv))

                itf_pow[n] = itf_pow_temp

            for n_u in range(N):
                common_rate_vec[n_u] = np.log2(1 + (cm_pow[n_u]/(itf_pow[n_u] + tar_pow[n_u] + 1)))
                private_rate_vec[n_u] = np.log2(1 + (tar_pow[n_u]/(itf_pow[n_u] + 1)))

            new_rate = np.sum(private_rate_vec) + np.amin(common_rate_vec)
        else:
            demo_flag = 0
            demo_action = None
            demo_action_delta = None
            new_rate = 0
            self.current_bfv = current_bfv

        tx_bfv = current_bfv.T
        tx_bfv = tx_bfv[...,np.newaxis]
        rx_bfv = np.random.uniform(1,1,size=(N,Nr,1))
        cm_pow = np.zeros(N)
        tar_pow = np.zeros(N)
        itf_pow = np.zeros(N)
        common_rate_vec = np.zeros((1,N))
        private_rate_vec = np.zeros((1,N))
        for n in range(N):
            H_n = H[[n],:]
            cm_bfv_ = tx_bfv[0,:]
            rx_bfv_ = np.asmatrix(rx_bfv[n,:])
            tx_bfv_ = tx_bfv[n+1,:]

            common_signal = rx_bfv_.H @ H_n @ cm_bfv_
            private_signal = rx_bfv_.H @ H_n @ tx_bfv_

            cm_pow[n] = np.square(np.absolute(common_signal))
            tar_pow[n] = np.square(np.absolute(private_signal))
            itf_pow_temp = 0.

            real_H = np.real(H_n).flatten()
            imag_H = np.imag(H_n).flatten()

            
            if n == 0:
                state_vec = np.concatenate((real_H, imag_H))
            else:
                state_vec = np.concatenate((state_vec, real_H, imag_H))

            for j in range(N):
                if j != n:
                    itf_tx_bfv = tx_bfv[j+1,:]
                    itf_signal = rx_bfv_.H @ H_n @ itf_tx_bfv
                    itf_pow_temp += np.square(np.absolute(itf_signal))

            itf_pow[n] = itf_pow_temp

        for n_u in range(N):
            common_rate_vec[0,n_u] = np.log2(1 + (cm_pow[n_u]/(itf_pow[n_u] + tar_pow[n_u] + 1)))
            private_rate_vec[0,n_u] = np.log2(1 + (tar_pow[n_u]/(itf_pow[n_u] + 1)))
        
        current_bfv_real = np.real(current_bfv).flatten()
        current_bfv_imag = np.imag(current_bfv).flatten()

        scale_ = np.amax(np.maximum(np.real(H), np.imag(H)))
        H_scaled = H * scale_
        for n in range(N):
            H_n = H_scaled[[n],:]
            real_H = np.real(H_n).flatten()
            imag_H = np.imag(H_n).flatten()
            current_bfv_real_c = np.real(current_bfv)[:,0].flatten()
            current_bfv_imag_c = np.imag(current_bfv)[:,0].flatten()
            if n == 0:
                state_attn = np.concatenate((real_H,imag_H,current_bfv_real_c,current_bfv_imag_c,common_rate_vec[0,n].flatten()))
                state_attn = state_attn[None,:]
            else:
                temp = np.concatenate((real_H,imag_H,current_bfv_real_c,current_bfv_imag_c,common_rate_vec[0,n].flatten()))
                temp = temp[None,:]
                state_attn = np.concatenate((state_attn, temp), axis=0)

        for n in range(N):
            H_n = H_scaled[[n],:]
            real_H = np.real(H_n).flatten()
            imag_H = np.imag(H_n).flatten()
            current_bfv_real_n = np.real(current_bfv)[:,n+1].flatten()
            current_bfv_imag_n = np.imag(current_bfv)[:,n+1].flatten()
            temp = np.concatenate((real_H,imag_H, current_bfv_real_n, current_bfv_imag_n, private_rate_vec[0,n].flatten()))
            temp = temp[None,:]
            state_attn = np.concatenate((state_attn, temp), axis=0)
        
        state_vec = np.concatenate((state_vec, current_bfv_real, current_bfv_imag,\
            common_rate_vec.flatten(), private_rate_vec.flatten()))

        # Determine the reward as the function of user's rate
        common_rate = np.amin(common_rate_vec)
        old_rate = (common_rate/N) + private_rate_vec
        reward = np.sum(old_rate)
        reward_new = torch.tensor([new_rate])
        # print('reward_new', reward_new)

        state_vec = state_vec[None,:]
        state_attn = state_attn[None,:]
        # print('state_vec',state_vec,state_vec.shape)
        self.env_step_counter = self.env_step_counter + 1
        if env_step_counter >= max_step:
            self.done_flag = True


        return state_attn, state_vec, reward_new, self.done_flag, demo_flag, demo_action_delta

if __name__ == "__main__":
    print("---------------------------------------")
    print(f"Generating Demonstration Replay")
    print("---------------------------------------")
    env = Env()
    max_timesteps = 1e6 # Number of transition tuples to generate
    state_dim = (N+1)*N*2+ N + (N+1)*Nt*2
    action_dim = (N+1)*Nt*2

    replay = ReplayBuffer(state_dim, action_dim, N, Nt) # Initialize the demonstration replay

    episode_timesteps = 0
    episode_counter = 0
    
    state, action, done = env.reset()
    episode_start = 1
    for t in range(int(max_timesteps)):
        
        episode_timesteps += 1
        # if t % 20 == 0:
        #     print('t',t)

        next_atten_state, next_state, reward, done, demo_flag, demo_action = env.step(state,action) 
        done_bool = float(done)

        replay.add(state, demo_action, next_atten_state, reward, episode_counter, episode_start, done_bool) # Save the transition tuple
        episode_start = 0
        action = demo_action

        state = next_atten_state

        if done:
            print(f"Accomplished episodes: {episode_counter}, number of stored transition tuples: {t+1}")
            episode_counter += 1
            state, action, done = env.reset()
            episode_start = 1
            replay.save() # Save the demonstration replay
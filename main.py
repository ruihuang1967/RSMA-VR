import numpy as np
import torch
import argparse
import os
from utils import ReplayBuffer
import DeepGRAIL
import math
import logging
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_printoptions(precision=6)

Nt = 6
Nr = 1
N = 4
episode_length = 50
eval_episode_length = episode_length
global_action_scale = 0.1

# Physical environment
dis_max = 10
dis_min = 4
freq = 60 * (10**9)
sol = 299792458
pathloss_op = 2.29
bandwidth = 10**9
noise_p = 3.9810717055e-21 * bandwidth
normalizer = 1/noise_p
height = 3.5

class Env: # Implementation of the wireless systems

    def __init__(self):
        self.reset()
        
    def reset(self):
        self.demo_freq = 1
        self.action_scale = global_action_scale
        self.mse_coefficient = 10
        self.env_step_counter = 0
        self.ptb_step_counter = 0
        self.max_step = 40
        self.done_flag = False
        self.current_bfv = np.zeros((Nt,N+1),dtype=np.complex128)

        # Locations of users
        x_cor = np.random.uniform(1, 4, N)
        y_cor = np.random.uniform(1, 4, N)
        hori_dis = np.sqrt(np.power(x_cor,2) + np.power(y_cor,2))
        distance_ = np.sqrt(np.power(hori_dis,2) + np.power(height,2))
        tangent_AOD = np.asmatrix(np.arctan(height/hori_dis)).T

        # Generate steering vectors for CSI
        AoD_user = tangent_AOD
        AoA_user = np.asmatrix(np.random.uniform(math.pi/4, math.pi/4, N)).T

        rx_sv_expand =  np.asmatrix(np.arange(0, Nr)).T
        rx_sv_angle = rx_sv_expand @ np.sin(AoA_user).T
        rx_sv = np.exp(-1j*rx_sv_angle)

        tx_sv_expand =  np.asmatrix(np.arange(0, Nt)).T

        tx_sv_angle = tx_sv_expand @ np.sin(AoD_user).T
        tx_sv = np.exp(-1j*tx_sv_angle)

        complex_gain = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(N, 2)).view(np.complex128)

        distance = np.asmatrix(distance_).T
        pathloss_ = np.sqrt(np.power((4*math.pi*distance*freq)/sol, -pathloss_op)) * np.sqrt(normalizer)

        # Generate CSIs for the users
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
        action = np.random.uniform(0.5,0.5,size=(Nt,N+1,2))
        action_power = (np.linalg.norm(action))**2
        tx_bfv_scale = np.sqrt(1/action_power)
        action = tx_bfv_scale * action
        
        new_state, _, _, _, _ = self.step(ini_state, action)
        return new_state, action, False

    def step(self, state, action): # MDP decision epoch
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

        H = copy.deepcopy(self.user_channel)

        demo_flag = 0
        demo_action_delta = None
        self.current_bfv = current_bfv

        # Determining SINR and rate
        tx_bfv = current_bfv.T
        tx_bfv = tx_bfv[...,np.newaxis]
        rx_bfv = np.random.uniform(1,1,size=(N,Nr,1))
        cm_pow = np.zeros(N)
        tar_pow = np.zeros(N)
        itf_pow = np.zeros(N)
        common_rate_vec = np.zeros((1,N))
        private_rate_vec = np.zeros((1,N))

        # Obtain the common and private rates
        for n in range(N):
            H_n = H[[n],:]
            cm_bfv_ = tx_bfv[0,:]
            rx_bfv_ = np.asmatrix(rx_bfv[n,:])
            tx_bfv_ = tx_bfv[n+1,:]

            # Received signal power
            common_signal = rx_bfv_.H @ H_n @ cm_bfv_
            private_signal = rx_bfv_.H @ H_n @ tx_bfv_

            cm_pow[n] = np.square(np.absolute(common_signal))
            tar_pow[n] = np.square(np.absolute(private_signal))
            itf_pow_temp = 0.

            real_H = np.real(H_n).flatten()
            imag_H = np.imag(H_n).flatten()


            for j in range(N):
                if j != n:
                    itf_tx_bfv = tx_bfv[j+1,:]
                    itf_signal = rx_bfv_.H @ H_n @ itf_tx_bfv
                    itf_pow_temp += np.square(np.absolute(itf_signal))

            itf_pow[n] = itf_pow_temp

        # Common and private rates
        for n_u in range(N):
            common_rate_vec[0,n_u] = np.log2(1 + (cm_pow[n_u]/(itf_pow[n_u] + tar_pow[n_u] + 1)))
            private_rate_vec[0,n_u] = np.log2(1 + (tar_pow[n_u]/(itf_pow[n_u] + 1)))
        
        current_bfv_real = np.real(current_bfv).flatten()
        current_bfv_imag = np.imag(current_bfv).flatten()

        # Construct state for attention critic
        for n in range(N):
            H_n = H[[n],:]
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
            H_n = H[[n],:]
            real_H = np.real(H_n).flatten()
            imag_H = np.imag(H_n).flatten()
            current_bfv_real_n = np.real(current_bfv)[:,n+1].flatten()
            current_bfv_imag_n = np.imag(current_bfv)[:,n+1].flatten()
            temp = np.concatenate((real_H,imag_H, current_bfv_real_n, current_bfv_imag_n, private_rate_vec[0,n].flatten()))
            temp = temp[None,:]
            state_attn = np.concatenate((state_attn, temp), axis=0)

        # Determine the reward as the function of user's rate
        common_rate = np.amin(common_rate_vec)
        old_rate = (common_rate/N) + private_rate_vec # In this example we simply split the common rate equally among the users
        reward = np.sum(old_rate)
        reward_new = torch.tensor([reward])

        # state_vec = state_vec[None,:]
        state_attn = state_attn[None,:]
        self.env_step_counter = self.env_step_counter + 1
        if env_step_counter >= max_step:
            self.done_flag = True

        return state_attn, reward_new, self.done_flag, demo_flag, demo_action_delta

    def eval_step(self, state, action, lagarange, FP_opt):
        if self.env_step_counter % 100 == 0:
            print('env step',self.env_step_counter)
        env_step_counter = self.env_step_counter
        max_step = self.max_step
        current_bfv = copy.deepcopy(self.current_bfv)
        # action = np.reshape(action, (Nt, N+1,2))
        action_delta = action[:,:,0] + 1j*action[:,:,1]
        current_bfv += self.action_scale * action_delta
        tx_bfv_power = (np.linalg.norm(current_bfv))**2
        if tx_bfv_power > 0:
            tx_bfv_scale = np.sqrt(1/tx_bfv_power)
            current_bfv = tx_bfv_scale * current_bfv
        tx_bfv_power = (np.linalg.norm(current_bfv))**2
        
        current_bfv_real = np.clip(np.real(current_bfv), -1, 1)
        current_bfv_imag = np.clip(np.imag(current_bfv), -1, 1)

        current_bfv = current_bfv_real + 1j*current_bfv_imag
        self.current_bfv = current_bfv
       
        ini_bf = current_bfv.T
        ini_bf = ini_bf[...,np.newaxis]
        # print('current_bfv', current_bfv.shape)

        H = copy.deepcopy(self.user_channel)
        demo_flag = 0
        demo_action = None
        demo_action_delta = None

        # Initialized tx and rx beamforming vectors
        tx_bfv = ini_bf
        rx_bfv = np.random.uniform(1,1,size=(N,Nr,1))

        tx_bfv_power = (np.linalg.norm(tx_bfv))**2
        if tx_bfv_power > 0:
            tx_bfv_scale = np.sqrt(1/tx_bfv_power)
            tx_bfv = tx_bfv_scale * tx_bfv

        # Determining SINR and rate
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

            # obtain effective channel: common message, private message, aggregate intference message
            common_signal = rx_bfv_.H @ H_n @ cm_bfv_
            private_signal = rx_bfv_.H @ H_n @ tx_bfv_

            cm_pow[n] = np.square(np.absolute(common_signal))
            tar_pow[n] = np.square(np.absolute(private_signal))
            itf_pow_temp = 0.

            real_H = np.real(H_n).flatten()
            imag_H = np.imag(H_n).flatten()


            for j in range(N):
                if j != n:
                    itf_tx_bfv = tx_bfv[j+1,:]
                    itf_signal = rx_bfv_.H @ H_n @ itf_tx_bfv
                    itf_pow_temp += np.square(np.absolute(itf_signal))

            itf_pow[n] = itf_pow_temp

        for n_u in range(N):
            common_rate_vec[0,n_u] = np.log2(1 + (cm_pow[n_u]/(itf_pow[n_u] + tar_pow[n_u] + 1)))
            private_rate_vec[0,n_u] = np.log2(1 + (tar_pow[n_u]/(itf_pow[n_u] + 1)))

        # print('common rate', common_rate_vec)
        # print('private rate', private_rate_vec)
        
        current_bfv_real = np.real(current_bfv).flatten()
        current_bfv_imag = np.imag(current_bfv).flatten()

        # Construct state vector: (CSI, beamforming vectors, achievable rates)
        for n in range(N):
            H_n = H[[n],:]
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
            H_n = H[[n],:]
            real_H = np.real(H_n).flatten()
            imag_H = np.imag(H_n).flatten()
            current_bfv_real_n = np.real(current_bfv)[:,n+1].flatten()
            current_bfv_imag_n = np.imag(current_bfv)[:,n+1].flatten()
            temp = np.concatenate((real_H,imag_H, current_bfv_real_n, current_bfv_imag_n, private_rate_vec[0,n].flatten()))
            temp = temp[None,:]
            state_attn = np.concatenate((state_attn, temp), axis=0)

        # Determine the reward as the function of user's rate
        common_rate = np.amin(common_rate_vec)
        old_rate = (common_rate/N) + private_rate_vec
        reward = np.sum(old_rate)
        reward_new = torch.tensor([reward])
        # print('reward_new', reward_new)

        state_attn = state_attn[None,:]
        self.env_step_counter = self.env_step_counter + 1
        if env_step_counter >= max_step:
            self.done_flag = True


        return state_attn, reward_new, self.done_flag

def eval_policy(policy, eval_episodes):
    eval_env = Env()

    avg_reward = 0.
    avg_max_rate = 0.
    avg_action_deviation = 0.
    avg_rate_vec = 0.
    eval_step_total = 0.

    for eval_counter in range(eval_episodes):
        state, action, done = eval_env.reset()
        eval_env.max_step = eval_episode_length

        eval_epsd_max = 0.
        while not done:

            action = policy.select_action(np.array(state))
            state_attn, reward_new, done, _, _ = eval_env.step(state, action)
            reward = reward_new.item()  

            avg_reward += reward
            eval_step_total += 1
            state = state_attn

            if reward>= eval_epsd_max:
                eval_epsd_max = reward


        avg_max_rate += eval_epsd_max    

    avg_reward /= eval_step_total
    avg_max_rate /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes, avg_agg_thrput: {avg_reward:.3f}, avg_max:{avg_max_rate:.2f}")
    print("---------------------------------------")

    return avg_reward, avg_max_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DeepGRAIL")            # Policy
    parser.add_argument("--csi_bound", default=0.20)                # CSI error bound (not implemented)
    parser.add_argument("--sample_num", default=100, type=int)     # Sample number
    parser.add_argument("--seed", default=0, type=int)              # Random seed
    parser.add_argument("--start_timesteps", default=1, type=int)   # Time steps for exploration. Please set it to a large value, e.g., 1e5, 
                                                                    # if you are not using the downloaded replay
    parser.add_argument("--eval_freq", default=50, type=int)        # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e7, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.3)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=512, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.95)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--w1", default=1e-3)                       # Weights of the deterministic policy graident (w1)
    parser.add_argument("--w2", default=1)                          # Weights of the imitation loss (w2)
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.2)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    total_sample_num = args.sample_num
    env = Env()
    rewards = []
    robust_loss_log = []
    actor_loss_log = []
    eval_thrput_log = []
    eval_max_log = []
    target_Q_log = []
    robust_loss_logger_rt = 0.
    actor_loss_logger_rt = 0.
    eval_falg = 0

    # Set seeds
    # env.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    
    state_dim = (N+1)*N*2+ N + (N+1)*Nt*2
    action_dim = (N+1)*Nt*2
    max_action = 1.0

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "w1": args.w1,
        "w2": args.w2,
        "action_scale": global_action_scale,
    }

    # Initialize policy
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = DeepGRAIL.Agent(**kwargs)

    # Create and load Demonstration Replay
    expert_replay = ReplayBuffer(state_dim, action_dim, N, Nt) # Demonstration replay
    expert_replay.load(n_step=5) # Load demonstration replay using n-step returns

    # Create and load Experience Replay
    agent_replay = ReplayBuffer(state_dim, action_dim, N, Nt) # Experience replay
    agent_replay.load_explore(n_step=5) # Load experience replay (optional, only use if you use the downloaded replays)

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_actor_loss = 0.
    episode_robust_loss = 0.
    episode_target_Q = 0.


    episode_counter = 0.
    state, action, done = env.reset()
    episode_start = 1
    log_step = 200
    
    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # if t == args.start_timesteps:                       # Please comment this if you're using the downloaded replay
        #     agent_replay.save_explore()                     # If you change the system setting, make sure to do the exploration again

        # if t < args.start_timesteps:                        # To save time, run this once to obtain the experience replay (i.e., exploration)
        #     action = np.random.uniform(-1,1,size=(Nt,N+1,2))
        #     action_power = (np.linalg.norm(action))**2
        #     tx_bfv_scale = np.sqrt(1/action_power)
        #     action = tx_bfv_scale * action
        # else:
        action = (
            policy.select_action(np.array(state))
            + np.random.normal(0, max_action * args.expl_noise, size=(Nt,N+1,2))
        ).clip(-max_action, max_action)
        action_power = (np.linalg.norm(action))**2
        tx_bfv_scale = np.sqrt(1/action_power)
        action = tx_bfv_scale * action

        next_state, reward, done, demo_flag, demo_action_delta\
                = env.step(state, action)
        done_bool = float(done)


        agent_replay.add(state, action, next_state, reward, episode_counter, episode_start, done_bool)
        episode_start = 0
        eval_falg = 1

        state = next_state

        if t >= args.start_timesteps:
            if t % 50 == 0:
                print('t', t)
            robust_loss_logger_rt, actor_loss_logger_rt, target_Q_logger_rt \
                = policy.train(agent_replay, expert_replay, args.batch_size)

            episode_robust_loss += robust_loss_logger_rt
            episode_actor_loss += actor_loss_logger_rt
            episode_target_Q += target_Q_logger_rt

        if done:
            # print('episode:', episode_counter, t)
            agent_replay.update_total_ind(n_step=5)
            episode_counter += 1
            state, action, done = env.reset()
            episode_start = 1
            # if episode_counter % 20 == 0:
              
        if t % log_step == 0 and t > 5:
            episode_num += 1
            episode_avg_robust_loss = round(episode_robust_loss/log_step, 5)
            episode_avg_actor_loss = round(episode_actor_loss/log_step, 5)
            episode_avg_target_Q = round(episode_target_Q/log_step, 3)

            robust_loss_log.append(episode_avg_robust_loss)
            actor_loss_log.append(episode_avg_actor_loss)
            target_Q_log.append(episode_avg_target_Q)
            if t % 10 == 0:
                with open("critic_loss_log.txt", "w") as output:
                    output.write(str(robust_loss_log))
                with open("actor_loss_log.txt", "w") as output:
                    output.write(str(actor_loss_log))
                with open("target_Q_log.txt", "w") as output:
                    output.write(str(target_Q_log))

            print('========================================')
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps}")
            print(f"Actor Loss: {episode_avg_actor_loss} Robust Loss: {episode_avg_robust_loss} Target Q: {episode_avg_target_Q}")
            print('========================================')

            episode_actor_loss = 0.
            episode_robust_loss = 0.
            episode_target_Q = 0.
        
        if t % args.eval_freq == 0 and eval_falg > 0:
            eval_reward, eval_max = eval_policy(policy, 50)
            
            eval_thrput_log.append(eval_reward)
            eval_max_log.append(eval_max)
            with open("eval_thrput_log.txt", "w") as output:
                output.write(str(eval_thrput_log))
            with open("eval_max_log.txt", "w") as output:
                output.write(str(eval_max_log))


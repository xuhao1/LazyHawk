
import matplotlib.pyplot as plt

import torch
from DQN import DQNwithState
from xplaneenv import XPlaneEnvCon, XPlaneEnv

from PyTorchRL.utils import *
from PyTorchRL.models.mlp_policy import Policy
from PyTorchRL.models.mlp_critic import Value
from PyTorchRL.core.trpo import trpo_step
from PyTorchRL.core.common import estimate_advantages
from PyTorchRL.core.agent import Agent
import time
import os
import pickle

import numpy as np

dtype = torch.float64
torch.set_default_dtype(dtype)
plt.rc('figure', figsize=(15, 8))


def work_dqn():
    xplaneenv = XPlaneEnv()
    xplaneenv.update()
    DQN = DQNwithState(8, 7, xplaneenv, batch_size=4)
    DQN.main_loop(20, show_debug=False)
    plt.show()

def work_trpo(params):
    state_dim = 11
    action_dim = 2
    policy_net = Policy(state_dim, action_dim, log_std=-1)

    value_net = Value(state_dim)

    # policy_net, value_net, running_state = pickle.load(open('logs/learned_models/glider_long_trpo_25.p', "rb"))
    # policy_net, value_net, running_state = pickle.load(open('learned_models/glider_trpoalt.p', "rb"))

    device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')

    policy_net.to(device)
    value_net.to(device)

    xplaneenv = XPlaneEnvCon(baseline=False)
    running_state = ZFilter((state_dim,), clip=5)

    agent = Agent(xplaneenv, policy_net, device, running_state=running_state, render=False,
                  num_threads=1)

    plt.figure('Average Indicate Climb Rate')
    aver_vel_ind = []
    heightd = []
    ted = []
    staytime = []
    base_folder = f"logs/TRPO_SLongTE_{time.strftime('%Y-%m-%d_%H_%M', time.localtime()) }"
    os.mkdir(base_folder)
    def update_params(batch):
        states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
        actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
        with torch.no_grad():
            values = value_net(states)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, params['gamma'], params['tau'], device)

        """perform TRPO update"""
        trpo_step(policy_net, value_net, states, actions, returns, advantages, params['max_kl'], params['damping'],
                  params['l2_reg'])

    for i_iter in range(params['max_iter']):
        """generate multiple trajectories that reach the minimum batch_size"""

        print("Collecting samples.....")
        batch, log = agent.collect_samples(params['min_batch_size'], params['single_period'])
        t0 = time.time()
        update_params(batch)
        t1 = time.time()
        
        print('{}\ Stay Time, {:.1f} T_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}\tD_H {:.1f} D_TE {:.1f}'.format(
                i_iter,log['tstay'], log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward'], log['delta_height'], log['delta_te']))
            
        if i_iter % params['log_interval'] == 0:
            aver_vel_ind.append(log['avg_reward']/1000)
            heightd.append(log['delta_height'])
            ted.append(log['delta_te'])
            staytime.append(log['tstay'])

            plt.figure('Average Indicate Climb Rate')
            plt.clf()

            plt.plot(aver_vel_ind, label='Average Indicate Climb Rate')
            plt.grid(which = "both")
            plt.legend()
            plt.savefig(f"{base_folder}/SLTE_average_climb_{time.strftime('%Y-%m-%d_%H_%M', time.localtime()) }.png")
            plt.savefig(f"{base_folder}/SLTE_average_climb.png")

            plt.figure("Altitude Changes")
            plt.clf()
            plt.plot(heightd, label="Altitude Changes")
            plt.grid(which = "both")
            plt.legend()
            plt.savefig(f"{base_folder}/Dalt_{time.strftime('%Y-%m-%d_%H_%M', time.localtime()) }.png")
            plt.savefig(f"{base_folder}/Dalt.png")


            plt.figure("Total Energy Changes")
            plt.clf()
            plt.plot(ted, label="Total Energy Changes")
            plt.grid(which = "both")
            plt.legend()
            plt.savefig(f"{base_folder}/TRPO_SLTE_{time.strftime('%Y-%m-%d_%H_%M', time.localtime()) }.png")
            plt.savefig(f"{base_folder}/TRPO_SLTE.png")


            plt.figure("Stay Time")
            plt.clf()
            plt.plot(staytime, label="Stay Time")
            plt.grid(which = "both")
            plt.legend()
            plt.savefig(f"{base_folder}/TRPO_StayTime_{time.strftime('%Y-%m-%d_%H_%M', time.localtime()) }.png")
            plt.savefig(f"{base_folder}/TRPO_StayTime_SLTE.png")

            print("Save traj")
            arr = np.array([log['trajx'], log['trajy'], log['trajz']] )
            arr = np.transpose(arr)
            np.savetxt(f"{base_folder}/traj_{time.strftime('%Y-%m-%d_%H_%M', time.localtime()) }.csv", arr, delimiter=",")
            # plt.pause(0.1)

        if params['save_model_interval'] > 0 and (i_iter+1) % params['save_model_interval'] == 0:
            print("save model")
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net, running_state),
                        open(f'{base_folder}/glider_super_long_trpo_{i_iter}.p', 'wb'))
            to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()
    plt.show()

trpo_long_defaults = {
        "gamma":0.99,
        "max_kl":1e-2,
        "damping":1e-2,
        "tau":0.95,
        "max_iter":1000,
        "min_batch_size": 2048*4,
        "log_interval":2,
        "save_model_interval":2,
        "l2_reg":1e-3,
        "single_period":1024
    }

trpo_short_defaults = {
        "gamma":0.99,
        "max_kl":1e-2,
        "damping":1e-2,
        "tau":0.95,
        "max_iter":1000,
        "min_batch_size": 1024,
        "log_interval":2,
        "save_model_interval":2,
        "l2_reg":1e-3,
        "single_period":128
    }
if __name__ == "__main__":
    print("Welcom to XPLANE REINFORCE")

    work_trpo(trpo_short_defaults)





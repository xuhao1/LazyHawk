
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

dtype = torch.float64
torch.set_default_dtype(dtype)
plt.rc('figure', figsize=(15, 8))


def work_dqn():
    xplaneenv = XPlaneEnv()
    xplaneenv.update()
    DQN = DQNwithState(8, 7, xplaneenv, batch_size=4)
    DQN.main_loop(20, show_debug=True)
    plt.show()

def work_trpo(params):
    state_dim = 11
    action_dim = 2
    policy_net = Policy(state_dim, action_dim, log_std=-1)

    value_net = Value(state_dim)

    policy_net, value_net, running_state = pickle.load(open('learned_models/glider_trpo.p', "rb"))

    device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')

    policy_net.to(device)
    value_net.to(device)

    xplaneenv = XPlaneEnvCon()
    running_state = ZFilter((state_dim,), clip=5)

    agent = Agent(xplaneenv, policy_net, device, running_state=running_state, render=False,
                  num_threads=1)

    plt.figure('Average Indicate Climb Rate')
    aver_vel_ind = []
    heightd = []
    ted = []

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
        batch, log = agent.collect_samples(params['min_batch_size'])
        t0 = time.time()
        update_params(batch)
        t1 = time.time()

        if i_iter % params['log_interval'] == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}\tD_H {:.1f} D_TE {:.1f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward'], log['delta_height'], log['delta_te']))
            aver_vel_ind.append(log['avg_reward']/1000)
            heightd.append(log['delta_height'])
            ted.append(log['delta_te'])

            plt.figure('Average Indicate Climb Rate')
            plt.clf()

            plt.plot(aver_vel_ind, label='Average Indicate Climb Rate')
            plt.grid(which = "both")
            plt.legend()
            # plt.savefig(f"TRPO_TE_average_climb_{time.strftime('%Y-%m-%d_%H_%M', time.localtime()) }.pdf")
            plt.savefig(f"TRPO_TE_average_climb.pdf")

            plt.figure("Altitude Changes")
            plt.clf()
            plt.plot(heightd, label="Altitude Changes")
            plt.grid(which = "both")
            plt.legend()
            # plt.savefig(f"TRPO_TE_dalt_{time.strftime('%Y-%m-%d_%H_%M', time.localtime()) }.pdf")
            plt.savefig(f"TRPO_TE_dalt.pdf")


            plt.figure("Total Energy Changes")
            plt.clf()
            plt.plot(ted, label="Total Energy Changes")
            plt.grid(which = "both")
            plt.legend()
            # plt.savefig(f"TRPO_TE_{time.strftime('%Y-%m-%d_%H_%M', time.localtime()) }.pdf")
            plt.savefig(f"TRPO_TE.pdf")

            # plt.pause(0.1)

        if params['save_model_interval'] > 0 and (i_iter+1) % params['save_model_interval'] == 0:
            print("save model")
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net, running_state),
                        open('learned_models/glider_trpo.p', 'wb'))
            to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()
    plt.show()

trpo_defaults = {
        "gamma":0.99,
        "max_kl":1e-2,
        "damping":1e-2,
        "tau":0.95,
        "max_iter":50,
        "min_batch_size": 1024,
        "log_interval":1,
        "save_model_interval":1,
        "l2_reg":1e-3
    }
if __name__ == "__main__":
    print("Welcom to XPLANE REINFORCE")

    work_trpo(trpo_defaults)





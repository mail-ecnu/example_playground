import os
import random
from collections import namedtuple
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import pommerman
from pommerman import agents

writer = SummaryWriter("./log_2/")
global_step = 0

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
reward_list = []


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.nseq1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.nseq2 = nn.Sequential(
            nn.Linear(1740, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )

    def forward(self, obs):
        obs = obs.reshape((-1, 1464))
        batch_size = obs.shape[0]
        obs_board_cent = torch.cat([obs[:, :121], obs[:, 366:487], obs[:, 732:853], obs[:, 1098:1219]], dim=1).reshape(
            (batch_size, 4, 11, 11))
        obs_bbs_cent = torch.cat([obs[:, 121:242], obs[:, 487:608], obs[:, 853:974], obs[:, 1219:1340]], dim=1).reshape(
            (batch_size, 4, 11, 11))
        obs_bl_cent = torch.cat([obs[:, 242:363], obs[:, 608:729], obs[:, 974:1095], obs[:, 1340:1461]], dim=1).reshape(
            (batch_size, 4, 11, 11))
        obs_other = torch.cat([obs[:, 363:366], obs[:, 729:732], obs[:, 1095:1098], obs[:, 1461:]], dim=1)
        cnn_output = torch.cat(
            [self.nseq1(obs_board_cent).reshape(batch_size, -1), self.nseq1(obs_bbs_cent).reshape(batch_size, -1),
             self.nseq1(obs_bl_cent).reshape(batch_size, -1)], dim=1)
        seq2_input = torch.cat([obs_other, cnn_output], dim=1)
        # board = obs[:,:,:363].reshape((-1, 12,11,11))
        # board = self.nseq1(board)
        # board = board.reshape((obs.size(0), -1))
        # print(board.shape)
        # obs = torch.cat([board, obs[:, 363:366]], dim=1)
        return self.nseq2(seq2_input)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.999
        self.obs_width = 11
        self.lr = 0.001
        self.batch_size = 256

        self.policy_net = Net().to(self.device)
        self.target_net = Net().to(self.device)
        if os.path.exists("model_2.pth"):
            self.policy_net.load_state_dict(torch.load("model_2.pth"))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(14000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        # policy net
        state_batch = torch.cat(batch.state).float()
        action_batch = torch.cat(batch.action).reshape((self.batch_size, -1))

        state_action_values = self.policy_net(state_batch).gather(dim=0, index=action_batch)
        state_action_values = state_action_values.reshape(self.batch_size)

        # target net
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).float()
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        reward_batch = torch.cat(batch.reward)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # print(expected_state_action_values)
        # calculate Q value loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        global writer
        global global_step

        writer.add_scalar("reward", torch.mean(expected_state_action_values).item(), global_step)
        writer.add_scalar("loss", loss.item(), global_step)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is None:
                print(param, "============", sep="\n")
            else:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class DQNAgent(agents.BaseAgent):
    def __init__(self, model):
        super(DQNAgent, self).__init__()
        self.model = model
        self.random = agents.SimpleAgent()
        self.obs_fps = [torch.zeros(366, device=self.model.device), torch.zeros(366, device=self.model.device),
                        torch.zeros(366, device=self.model.device)]

    def translate_obs(self, o):
        obs_width = self.model.obs_width

        board = o['board'].copy()
        agents = np.column_stack(np.where(board > 10))

        for i, agent in enumerate(agents):
            agent_id = board[agent[0], agent[1]]
            if agent_id not in o['alive']:  # < this fixes a bug >
                board[agent[0], agent[1]] = 0
            else:
                board[agent[0], agent[1]] = 11

        obs_radius = obs_width // 2
        pos = np.asarray(o['position'])

        # board
        board_pad = np.pad(board, (obs_radius, obs_radius), 'constant', constant_values=1)
        self.board_cent = board_cent = board_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

        # bomb blast strength
        bbs = o['bomb_blast_strength']
        bbs_pad = np.pad(bbs, (obs_radius, obs_radius), 'constant', constant_values=0)
        self.bbs_cent = bbs_cent = bbs_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

        # bomb life
        bl = o['bomb_life']
        bl_pad = np.pad(bl, (obs_radius, obs_radius), 'constant', constant_values=0)
        self.bl_cent = bl_cent = bl_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

        return np.concatenate((
            board_cent, bbs_cent, bl_cent,
            o['blast_strength'], o['can_kick'], o['ammo']), axis=None)

    def act(self, observation, action_space):
        obs = self.translate_obs(observation)
        obs = torch.from_numpy(obs).float().to(self.model.device)
        self.obs_fps.append(obs)
        obs = torch.cat(self.obs_fps[-4:])
        sample = random.random()
        if sample > 1000.0 / (global_step + 0.1):
            re_action = self.model.policy_net(obs).argmax().item()
            return re_action
        else:
            return self.random.act(observation, action_space)

    def episode_end(self, reward):
        global reward_list
        reward_list.append(reward)


if __name__ == "__main__":
    # train
    a = DQNAgent(DQN())
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        a
    ]
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    num_episodes = 100001
    for i_episode in range(num_episodes):

        # Initialize the environment and state
        state = env.reset()
        state_list = [torch.zeros(366, device=a.model.device), torch.zeros(366, device=a.model.device),
                      torch.zeros(366, device=a.model.device), torch.zeros(366, device=a.model.device),
                      torch.tensor(a.translate_obs(state[3]), device=a.model.device)]
        for t in count():
            # Select and perform an action
            # env.render()
            action = env.act(state)
            next_state, reward, done, info = env.step(action)
            if 13 in state[0]["alive"]:
                action = torch.tensor([action[3]], device=a.model.device)
                reward = torch.tensor([reward[3]], device=a.model.device)
                state = torch.tensor(a.translate_obs(state[3]), device=a.model.device)
                state_list.append(state)

                # Observe new state
                if not done:
                    next_state_ = torch.tensor(a.translate_obs(next_state[3]), device=a.model.device)
                else:
                    next_state_ = torch.zeros(366, device=a.model.device)

                state_list.append(next_state_)
                # Store the transition in memory
                push_state = torch.cat(state_list[-5:-1])
                push_next_state = torch.cat(state_list[-4:])
                a.model.memory.push(push_state, action, push_next_state, reward)
                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                a.model.learn()
                if done:
                    break
            else:
                a.model.learn()
                break
        global_step += 1
        # Update the target network, copying all weights and biases in DQN
        if i_episode % 10 == 0:
            print("第{}次-第{}次正在训练".format(i_episode + 1, i_episode + 10))
            a.model.target_net.load_state_dict(a.model.policy_net.state_dict())
            torch.save(a.model.target_net.state_dict(), "model_2.pth")

    print('Complete')

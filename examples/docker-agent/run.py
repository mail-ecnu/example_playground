"""Implementation of a simple deterministic agent using Docker."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pommerman import agents
from pommerman.runner import DockerAgentRunner

import random

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # self.nseq1 = nn.Sequential(
        #     nn.Conv2d(12,16,kernel_size=3),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16,32,kernel_size=5),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32,64,kernel_size=3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        self.nseq2 = nn.Sequential(
            nn.Linear(1464 ,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,6),
        )

    def forward(self, obs):
        obs = obs.reshape((-1,1464))
        # board = obs[:,:,:363].reshape((-1, 12,11,11))
        # board = self.nseq1(board)
        # board = board.reshape((obs.size(0), -1))
        # print(board.shape)
        # obs = torch.cat([board, obs[:, 363:366]], dim=1)
        return self.nseq2(obs)


class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.obs_width = 11
        import os
        if os.path.exists("../model.pth"):
            self.model.load_state_dict(torch.load("model.pth"))
        self._agent = agents.SimpleAgent()
        self.obs_fps = [torch.zeros(366),torch.zeros(366),torch.zeros(366)]

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def translate_obs(self, o):
        obs_width = self.obs_width

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
        obs = torch.from_numpy(obs).float().to(self.device)
        self.obs_fps.append(obs)
        obs = torch.cat(self.obs_fps[-4:])
        sample = random.random()
        if sample > 0.1:
            re_action = self.model(obs).argmax().item()
            return re_action
        else:
            return self._agent.act(observation, action_space)

    def episode_end(self, reward):
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()


def main():
    '''Inits and runs a Docker Agent'''
    agent = MyAgent()
    agent.run()


if __name__ == "__main__":
    main()

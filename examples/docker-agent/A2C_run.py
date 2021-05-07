"""Implementation of a simple deterministic agent using Docker."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pommerman import agents
from pommerman.runner import DockerAgentRunner

import random

from A2C.model import *


class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        self.model = A2CNet(gpu=False)
        import os
        if os.path.exists("A2C/convrnn-s.weights"):
            self.model.load_state_dict(torch.load("A2C/convrnn-s.weights", map_location='cpu'))
        self.agent = Leif(self.model)

    def act(self, observation, action_space):
        return self.agent(torch.from_numpy(observation).float().to('cpu'), action_space)



if __name__ == "__main__":
    agent = MyAgent()
    agent.run()

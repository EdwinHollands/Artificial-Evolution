# GET PYTORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from Building_Blocks import MLP

# USE GPU IF POSS
device        = 'cuda' if torch.cuda.is_available() else 'cpu'

# !!!!!!!!!! activate the venv first
# MIGHT NEED TO COPY TO TERMINA: C:\Users\echol\Documents\Coding\AI-practice\gpt_env\Scripts\activate

#PARAMETERS-----------------------
SIBLINGS = 3
CHILDHOOD_YEARS = 10
ADULTHOOD_YEARS = 5
MEMORY_LENGTH = 2
 
#PRISONERS' DILEMMA PAYOFF MATRIX----------------
# 0 = cooperate, 1 = defect
PAYOFF = {
    (0, 0): 3,  # mutual cooperation
    (0, 1): 0,  # sucker
    (1, 0): 5,  # exploit
    (1, 1): 1,  # mutual defection
}

class Brain(nn.Module):
    BLOCK_TYPES = {
        'MLP' : MLP,
    }

    def __init__(self, structure):
        super().__init__()
        self.structure = structure
        self.blocks = nn.ModuleList([
            self.BLOCK_TYPES[block_type](config)
            for block_type, config in structure
        ])
    def forward(self, inputs):
        out = inputs
        for block in self.blocks:
            out = block(out)
        return out
 
class Animal:
    id = 0
    def __init__(self, structure, memory_length, prediliction, skepticism, rumination, learn_rate=1e-3, generation=0):
        self.id = Animal.id
        Animal.id +=1
        self.generation = generation
        self.structure = structure
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.brain = Brain(structure).to(self.device)
        self.optimiser = torch.optim.Adam(self.brain.parameters(), lr=learn_rate)
        self.fitness = 0
        self.reputation = 0.5
        self.memory_length = memory_length 
        self.prediliction = prediliction #starting rate of defection
        self.skepticism = skepticism #assumed opponent rate of defection
        self.memory = torch.tensor(
            [[0, 0, self.prediliction, self.skepticism]] * memory_length, #(self.rep, opp.rep, intention, assumption)
            dtype=torch.float32,
            device=self.device
        )
        self.experience = []
        self.rumination = rumination
        self.round = 0  #actual games played

    def meet(self, opp_rep):
        impression = torch.tensor(
            [[self.reputation, opp_rep, self.prediliction, self.skepticism]],
            dtype=torch.float32, device=self.device
        )
        self.memory = torch.cat([self.memory, impression])

    def remember(self, my_a, opp_a):
        self.memory[-1, 2] = my_a
        self.memory[-1, 3] = opp_a
        self.round += 1
        self.reputation = (self.reputation+1-my_a)/(self.round+1)


    def forget(self):
        self.experience.append(self.memory.clone())
        self.memory = torch.tensor(
            [[0, 0, self.prediliction, self.skepticism]] * self.memory_length,
            dtype=torch.float32,
            device=self.device
        )
        self.round = 0

    def decide(self):
        self.brain.eval()
        window = self.memory[self.round : self.round + self.memory_length]
        with torch.no_grad():
            logits = self.brain(window.flatten())
            probs  = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
    
    def learn(self):
        if not self.experience:
            return
        self.brain.train()
        inputs, actions, weights = [],[],[]
        for game in self.experience:
            total = sum(
                PAYOFF[(int(game[t, 2].item()), int(game[t, 3].item()))]
                for t in range(self.memory_length, len(game))
            )
            for t in range(self.memory_length, len(game)):
                window = game[t-self.memory_length:t]
                inputs.append(window.flatten())
                actions.append(int(game[t, 2].item()))
                weights.append(total)
        x = torch.stack(inputs)                                              # (T, 4*memory_length)
        y = torch.tensor(actions, dtype=torch.long,    device=self.device)  # (T)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device)
        if w.std() > 0:
            w = (w - w.mean()) / (w.std())
        else:
            return
        for _ in range(self.rumination):
            self.optimiser.zero_grad()
            logits           = self.brain(x)                 # (T, n_actions)
            log_probs        = F.log_softmax(logits, dim=-1) # (T, n_actions)
            action_log_probs = log_probs[range(len(y)), y]   # (T,)
            loss             = -(action_log_probs * w).mean()
            loss.backward()
            self.optimiser.step()

    
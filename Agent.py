# GET PYTORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import copy
from Building_Blocks import MLP

# USE GPU IF POSS
device        = 'cuda' if torch.cuda.is_available() else 'cpu'

# !!!!!!!!!! activate the venv first
# MIGHT NEED TO COPY TO TERMINA: C:\Users\echol\Documents\Coding\AI-practice\gpt_env\Scripts\activate

#PARAMETERS-----------------------
TRIBE_SIZE = 4 #>2 for evolution
TRIBES = 10
POPULATION_SIZE = TRIBES * TRIBE_SIZE
MIN_ROUNDS = 4
MAX_ROUNDS = 6
CHILDHOOD_YEARS = 5
ADULTHOOD_YEARS = 5
MEMORY_LENGTH = 2
INPUT_DIM = 4 * MEMORY_LENGTH +2
INITIAL_STRUCTURE = [['MLP', [INPUT_DIM, 2]]]
INITIAL_RUMINATION = 10
INITIAL_LEARN_RATE = 1e-3

#PRISONERS' DILEMMA PAYOFF MATRIX----------------
# 0 = cooperate, 1 = defect
PAYOFF = {
    (0, 0): 1,  # mutual cooperation
    (0, 1): -2,  # sucker
    (1, 0): 3,  # exploit
    (1, 1): -1,  # mutual defection
}

#THE MACHINE LEARNING BRAIN OF THE ANIMAL ----------------------------
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
 
 #THE ANIMALS WHO WILL COMPETE ---------------------------------------
class Animal:
    id = 0
    def __init__(self, structure, memory_length, prediliction=0.5, skepticism=0.5, rumination=10, learn_rate=1e-3, generation=0):
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

    def remember(self, my_a, opp_a, opp_rep):
        new_row = torch.tensor(
            [[self.reputation, opp_rep, my_a, opp_a]],
            dtype=torch.float32, device=self.device
        )
        self.memory = torch.cat([self.memory, new_row])
        self.round += 1
        self.reputation = (self.round * self.reputation + 1 - my_a) / (self.round + 1)

    def process(self):
        self.experience.append(self.memory.clone())
        self.memory = torch.tensor(
            [[0, 0, self.prediliction, self.skepticism]] * self.memory_length,
            dtype=torch.float32,
            device=self.device
        )
        self.round = 0

    def decide(self, opp_rep):
        past = self.memory[-self.memory_length:].flatten()
        current = torch.tensor(
            [self.reputation, opp_rep],
            dtype=torch.float32, device=self.device
        )
        flat = torch.cat([past, current])  
        self.brain.eval()
        with torch.no_grad():
            logits = self.brain(flat)
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
                past = game[t-self.memory_length:t].flatten()
                current = game[t, :2]
                inputs.append(torch.cat([past, current]))
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

    def score(self):
        self.fitness = sum(
            PAYOFF[(int(game[t, 2].item()), int(game[t, 3].item()))]
            for game in self.experience
            for t in range(self.memory_length, len(game))
        )

# THE POPULATION ------------------------------------------------
class Population:
    def __init__(self, tribe_size, tribes, structure, memory_length,
                 prediliction=0.5, skepticism=0.5, rumination=10,
                 learn_rate=1e-3):

        self.tribe_size = tribe_size
        self.tribes = tribes
        self.size        = tribe_size * tribes
        self.generation  = 0
        self.structure   = structure
        self.memory_length = memory_length
        self.prediliction  = prediliction
        self.skepticism    = skepticism
        self.rumination    = rumination
        self.learn_rate    = learn_rate
        self.animals = [
            Animal(structure, memory_length, prediliction,
                   skepticism, rumination, learn_rate)
            for _ in range(tribe_size * tribes)
        ]
        self.history = []

    def childhood(self):
        animals = self.animals.copy()
        np.random.shuffle(animals)

        for i in range(0, len(animals), self.tribe_size):
            tribe = animals[i : i + self.tribe_size]
            for j in range(len(tribe)):
                for k in range(j + 1, len(tribe)):
                    for _ in range(CHILDHOOD_YEARS):
                        compete(tribe[j], tribe[k])

        for animal in self.animals:
            animal.learn()

    def adulthood(self):
        reps = [a.reputation for a in self.animals]
        print(f"  Rep range at adulthood start: {min(reps):.2f} - {max(reps):.2f}")
        for _ in range(ADULTHOOD_YEARS):
            animals = self.animals.copy()
            animals = sorted(animals, 
                key=lambda a: a.reputation, 
                reverse=True) # sort by rep
            for i in range(0, len(animals) - 1, 2):
                compete(animals[i], animals[i + 1])

    def score(self):
        for animal in self.animals:
            animal.score()

    def record(self):
        self.history.append({
            'generation'  : self.generation,
            'fitnesses'   : [a.fitness for a in self.animals],
            'reputations' : [a.reputation for a in self.animals],
        })

    def reproduce(self):
        pass

    def reset_generation(self):
        for animal in self.animals:
            animal.experience = []
            animal.fitness    = 0

    def step(self, report=False):
        self.childhood()
        self.adulthood()
        self.score()
        self.record()
        self.reproduce()
        self.reset_generation()
        self.generation += 1
        if report:
            fitnesses   = self.history[-1]['fitnesses']
            reputations = self.history[-1]['reputations']
            print(f"Gen {self.generation:3d} | "
                f"fitness  mean={np.mean(fitnesses):.1f}  "
                f"min={np.min(fitnesses):.1f}  "
                f"max={np.max(fitnesses):.1f}  | "
                f"coop  mean={np.mean(reputations):.2f}  "
                f"min={np.min(reputations):.2f}  "
                f"max={np.max(reputations):.2f}")

    def run(self, n_generations, report=False):
        for _ in range(n_generations):
            self.step(report=report)
        return self.history

    def plot(self):
        pass


# THE GAME ------------------------------------------------------------------
def compete(animal_a, animal_b):
    rounds = np.random.randint(MIN_ROUNDS, MAX_ROUNDS + 1)    
    for _ in range(rounds):
        a = animal_a.decide(animal_b.reputation)
        b = animal_b.decide(animal_a.reputation)
        animal_a.remember(a, b, animal_b.reputation)
        animal_b.remember(b, a, animal_a.reputation) 
    animal_a.process()
    animal_b.process()
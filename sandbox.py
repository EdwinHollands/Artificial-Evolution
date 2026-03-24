# GET PYTORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import copy
from Building_Blocks import MLP
from Agent import Brain, Animal, Population, compete

# USE GPU IF POSS
device        = 'cuda' if torch.cuda.is_available() else 'cpu'

#PARAMETERS-----------------------
INITIAL_MEMORY_LENGTH = 1
INPUT_DIM = 4 * INITIAL_MEMORY_LENGTH + 2
INITIAL_STRUCTURE = [['MLP', [INPUT_DIM, 2]]]
INITIAL_PREDILICTION = 0.1
INITIAL_SKEPTICISM = 0.1
INITIAL_RUMINATION = 2
INITIAL_LEARN_RATE = 1e-3
TRIBE_SIZE = 4 #>2 for evolution
TRIBES = 10
POPULATION_SIZE = TRIBES * TRIBE_SIZE
MIN_ROUNDS = 4
MAX_ROUNDS = 6
CHILDHOOD_YEARS = 5
ADULTHOOD_YEARS = 5
GENERATIONS = 20


#PRISONERS' DILEMMA PAYOFF MATRIX----------------
# 0 = cooperate, 1 = defect
PAYOFF = {
    (0, 0): 1,  # mutual cooperation
    (0, 1): -2,  # sucker
    (1, 0): 3,  # exploit
    (1, 1): -1,  # mutual defection
}

# SANDBOX ---------------------------------------------------------
pop = Population(
    tribe_size   = TRIBE_SIZE,
    tribes      = TRIBES,
    structure     = INITIAL_STRUCTURE,
    memory_length = INITIAL_MEMORY_LENGTH,
    prediliction  = INITIAL_PREDILICTION,
    skepticism    = INITIAL_SKEPTICISM,
    rumination    = INITIAL_RUMINATION,
    learn_rate    = INITIAL_LEARN_RATE,
)
print(f"Population: {pop.size} animals, "
        f"{sum(p.numel() for p in pop.animals[0].brain.parameters())} params each")
pop.run(GENERATIONS, report=True)
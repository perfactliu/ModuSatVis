"""
mask agents at cut vertex to keep configurtion connected
"""
import copy

import numpy as np
import torch
from environments.warshall import warshall
# from environment import SatelliteEnv


def mask2(positions, agent):
    mask = [1]*48
    temp_positions = positions
    temp_positions.remove(tuple(agent.position))
    if warshall(temp_positions) != 0:
        mask = [0]*48
    return np.array(mask)



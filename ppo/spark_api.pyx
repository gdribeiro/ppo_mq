#!./venv/bin/python3

import tensorflow as tf
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import deque
import time
import random
from tqdm import tqdm
import os
import csv
import array


##########################################################################################
# Cython API
##########################################################################################

cdef public object createAgent(float* start_state, int minqos, int maxqos):
    state = []
    for i in range(8):
        state.append(start_state[i])
    
    return AgentEpisode(state, maxqos, minqos)


cdef public int infer(object agent , float* new_state):
    state = []
    for i in range(8):
        state.append(new_state[i])

    agent.change_state(state)

    action = agent.get_action()
    action = action % (ACTION_SPACE_SIZE/2) - 1

    return action

cdef public void finish(object agent, float* last_state):
    state = []
    for i in range(8):
        state.append(last_state[i])
    
    agent.finish(state)

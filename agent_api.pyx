#!./venv/bin/python3

from cpython cimport array

from ppo_agent import PPOAgentMQ


##########################################################################################
# Cython API
##########################################################################################

cdef public object createPPOAgent(float* start_state, int qosmin, int qosmax):
    state = []
    for i in range(8):
        state.append(start_state[i])
    
    return PPOAgentMQ(state, qosmax, qosmin)


cdef public int infer(object agent , float* observation):
    state = []
    for i in range(8):
        state.append(observation[i])

    action = agent.step(state)

    return action

cdef public void finish(object agent, float* last_state):
    state = []
    for i in range(8):
        state.append(last_state[i])
    
    agent.finish(state)
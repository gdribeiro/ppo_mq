#!./venv/bin/python3
# cython: language_level=3

##########################################################################################
# DQN Imports
##########################################################################################
import tensorflow as tf
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp

from tf_agents.specs import tensor_spec
from tf_agents.specs import array_spec
from tf_agents.specs import BoundedTensorSpec
from tf_agents.specs import TensorSpec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import tf_py_environment

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment

from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
import timeit

import reverb
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

import csv

# TESTS
import random


##########################################################################################
### Constants
LOGS_DIR =  '/tmp/'
CSV_FILE = LOGS_DIR + "/log_rewards.csv"
LOG_FILE = LOGS_DIR + "/log_general.log"
AGENT_FILE = LOGS_DIR + "/log_agent.csv"

MODEL_NAME = 'DQN-LSTM-FullBuffer'

GLOBAL_BUFFER_SIZE = 200
GLOBAL_EPSILON = 0.1    # 0.1 is much more stable!
GLOBAL_EPOCHS = 3       #3 10 - 3 is BEST
GLOBAL_GAMMA = 0.99
GLOBAL_BATCH = 2   # 2 small is better
GLOBAL_STEPS = 2 # 2 is BEST
##########################################################################################

# DQN Agent 
class Agent:

    def __init__(self, env):
        self.policy_fc_layers= (32,32,32,32,32)
        self.q_fc_layers = self.policy_fc_layers
        self.epsilon = GLOBAL_EPSILON
        self.gamma = GLOBAL_GAMMA
        self.epochs = GLOBAL_EPOCHS

        self.time_step_tensor_spec = tensor_spec.from_spec(env.time_step_spec())
        self.observation_tensor_spec = tensor_spec.from_spec(env.observation_spec())
        self.action_tensor_spec = tensor_spec.from_spec(env.action_spec())

        self.q_net = self.createQNet()
        self.optimizer = self.createOptimizer()
        self.train_step_counter = tf.Variable(0)

        self.q_agent = self.createQAgent()

        self.batch_size = GLOBAL_BATCH
        self.num_steps = GLOBAL_STEPS
        self.buffer_size = GLOBAL_BUFFER_SIZE
        self.reverb_server = None
        self.replay_buffer = self.createReplayBuffer()
        self.iterator = self.createBufferIterator()

        self._loss = 10000
        self._eval = False
        self._counter = 0

        with open(AGENT_FILE, mode='w') as agentLog:
            writer = csv.writer(agentLog)
            writer.writerow(['step', 'StepCounter', 'Loss'])

    def createQNet(self):
        q_net = q_rnn_network.QRnnNetwork(
            input_tensor_spec=self.observation_tensor_spec,
            action_spec=self.action_tensor_spec,
            input_fc_layer_params=None,
            output_fc_layer_params=self.q_fc_layers,
            lstm_size=(32,)
        )
        return q_net

    def createOptimizer(self):
        learning_rate = 3e-4
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        return optimizer

    def createQAgent(self):
        print('TimeStepSpec: {}\n'.format(self.time_step_tensor_spec))
        q_agent = dqn_agent.DqnAgent(
            time_step_spec=self.time_step_tensor_spec,
            action_spec=self.action_tensor_spec,
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter
        )
        q_agent.initialize()
        print('Q Network: {}\n'.format(q_agent._q_network.summary()))
        # Optimazation: Disable/Ebable Autograph
        # q_agent.train = common.function(q_agent.train, autograph=False)
        q_agent.train_step_counter.assign(0)
        return q_agent

    def createReplayBuffer(self):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec= self.q_agent.policy.trajectory_spec,
            batch_size=1,
            max_length=self.buffer_size
        )
        # print('Replay Buffer Data Spec: {}\n'.format(replay_buffer.data_spec))
        return replay_buffer

    def addToBuffer(self, last_time_step, last_action, current_time_step):
        traj = trajectory.from_transition(last_time_step, last_action, current_time_step)
        traj_batched = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), traj)
        # print('Adding to Buffer - Trajectory: {}\n'.format(traj_batched))
        self.replay_buffer.add_batch(traj_batched)

    def createBufferIterator(self):
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=50,
            num_steps=self.num_steps
        ).prefetch(self.batch_size)
        iterator = iter(dataset)
        return iterator

    def train(self, global_step):
        if not (self.replay_buffer.num_frames().numpy() % (self.num_steps * self.batch_size)):
            experience, unused_info = next(self.iterator)
            # experience = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), experience)
            self._loss, _ = self.q_agent.train(experience)

            with open(AGENT_FILE, mode='a+', newline='') as agentLog:
                writer = csv.writer(agentLog)
                writer.writerow([global_step, self.train_step_counter.numpy(), self._loss.numpy()])
            print('TrainStep: {},\t LOSS: {}\n'.format(self.train_step_counter.numpy(), self._loss.numpy()))


    def getAction(self, time_step):
        policy_state = self.q_agent.policy.get_initial_state(batch_size=1)
        action = self.q_agent.policy.action(time_step, policy_state)

        # print('PolicyStep: {}'.format(action))
        action = tf.nest.map_structure(lambda x: tf.squeeze(x, axis=[0]), action)
        # print('PolicyStepSqueezed: {}'.format(action))
        return action
    

class MqEnvironment(py_environment.PyEnvironment):

    def __init__(self, maxqos, minqos):
        self._observation_spec = TensorSpec(shape=(8,), dtype=tf.float32, name='observation')

        # self._action_spec = BoundedTensorSpec(shape=(), dtype=tf.int32, minimum=minqos, maximum=maxqos, name='action')
        self._action_spec = BoundedTensorSpec(shape=(), dtype=tf.int32, minimum=0, maximum=1, name='action')
        # self._action_spec = BoundedTensorSpec(shape=(), dtype=tf.int32, minimum=0, maximum=2, name='action')
        # self._action_spec = BoundedTensorSpec(shape=(), dtype=tf.int32, minimum=0, maximum=6, name='action')
        self._reward_spec = TensorSpec(shape=(), dtype=tf.float32, name='reward')
        self._discount_spec = TensorSpec(shape=(), dtype=tf.float32, name='discount')

        self._maxqos = maxqos
        self._minqos = minqos
        self._rewards = 0
        self._current_time_step = None
        self._action = None
        self._discount = tf.convert_to_tensor(GLOBAL_GAMMA, dtype=tf.float32)
        with open(CSV_FILE, mode='w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['step','thpt_glo', 'thpt_var', 'cDELAY', 'cTIMEP', 'RecSparkTotal', 'RecMQTotal', 'state', 'mem_use','reward'])

        self._max_cDELAY = 2000
        self._max_cTIMEP = 2000
        self._avg_thpt = 0
        self._avg_cDELAY = 0
        self._avg_cTIMEP = 0
        self._window_time = 2000
        self._max_thpt = 0



    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reward_spec(self):
        return self._reward_spec

    def discount_spec(self):
        return self._discount_spec

    def _reset(self):
        observation_zeros = tf.zeros((8,), dtype=tf.float32)
        reward = tf.convert_to_tensor(1, dtype=tf.float32)
        self._current_time_step = ts.transition(observation_zeros, reward=reward, discount=self._discount)
        self._current_time_step = ts.TimeStep(tf.convert_to_tensor(ts.StepType.FIRST), reward, self._discount, observation_zeros)
        return self._current_time_step

    def _step(self, action):
        self._action = action
        return self._current_time_step

    def mq_step(self, action, state, global_step):
        observation = tf.convert_to_tensor(state, dtype=tf.float32)
        reward = self.get_reward(observation, global_step)
        self._current_time_step = ts.transition(observation, reward=reward, discount=self._discount)
        self._action = action
        return self._current_time_step

    def get_reward(self, observation, global_step):
        # [total_throughput, thpt_variation, proc_t, sche_t, msgs_to_spark, msgs_in_gb, ready_mem, spark_thresh]
        thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use = observation.numpy()
        lst_thpt_glo, lst_thpt_var, lst_cDELAY, lst_cTIMEP, lst_RecSparkTotal, lst_RecMQTotal, lst_state, lst_mem_use = self.current_time_step().observation.numpy()
        r_thpt_glo, r_thpt_var, r_cDELAY, r_cTIMEP, r_RecSparkTotal, r_RecMQTotal, r_state, r_mem_use = np.zeros(8, dtype=np.float32)

        reward = self.reward_alpha(observation)
        # reward = self.reward_beta(observation)
        # reward = self.reward_gamma(observation)
        # reward = self.reward_function2(observation)
        
        self._rewards += reward
        print('** Reward: {}\n** Total Rewards: {}'.format(reward, self._rewards))

        with open(CSV_FILE, mode='a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([global_step, thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use, reward])
        return tf.convert_to_tensor(reward, dtype=tf.float32)

    def reward_gamma(self, observation): # Opotimization vars: thpt_glo,  cDELAY, cTIMEP, state
        thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, qosbase = observation.numpy()
        lst_thpt_glo, lst_thpt_var, lst_cDELAY, lst_cTIMEP, lst_RecSparkTotal, lst_RecMQTotal, lst_state, lst_qosbase = self.current_time_step().observation.numpy()

        reward = 0.0

        # keep alive if other rewards don't apply
        reward = 0.0001

        if cDELAY > self._window_time or cTIMEP > self._window_time:
            # Reward to lower the memory usage
            reward = self.r_state_lin_norm_cost(state)
            if cTIMEP > self._window_time * 1.25:
                reward = -1.0
            elif cDELAY > self._window_time * 4:
                reward = -1.0
            elif cDELAY <= lst_cDELAY and qosbase < state:
            # qosbse < state means data will be processed and delay will decrease
                reward = 1
                
        else:
            if thpt_glo > self._avg_thpt:
                r_cDELAY = self.r_cDELAY_lin_norm_Inverted_original(cDELAY)
                r_cTIMEP = self.r_cTIMEP_lin_norm_Inverted_original(cTIMEP)
                r_state  = self.r_state_lin_norm_Inverted(state)
                rewards_p = np.array([r_cDELAY, r_cTIMEP, r_state], dtype=np.float32)
                weights = np.array([1, 1, 1])
                reward = np.average(rewards_p, weights=weights)


        self._avg_thpt = (self._avg_thpt + thpt_glo) / 2

        reward = np.round(reward * 1000) / 1000
        reward = np.clip(reward, a_min=-1.0, a_max=1.0)
        return reward

    def r_cDELAY_lin_norm_Inverted_original(self, cDELAY):
        # r_cDELAY = 1 - (cDELAY / self._max_cDELAY)
        r_cDELAY = (cDELAY / self._max_cDELAY)
        r_cDELAY = np.clip(r_cDELAY, a_min=0.0, a_max=1.0)
        return r_cDELAY

    def r_cTIMEP_lin_norm_Inverted_original(self, cTIMEP):
        if cTIMEP > 0:
            # r_cTIMEP = 1 - (cTIMEP / self._window_time)
            r_cTIMEP = (cTIMEP / self._window_time)
            r_cTIMEP = np.clip(r_cTIMEP, a_min=0.0, a_max=1.0)
            return r_cTIMEP
        else:
            return 0

    def r_state_lin_norm_cost(self, state):
        r_state = 0.0
        # min_range = self._minqos / 2
        min_range = self._minqos
        # max_range = self._maxqos / 2
        max_range = self._maxqos
        if state > self._maxqos:
            r_state = -1.0
        else:
            # r_state = (state) / (self._maxqos/2)
            # r_state = r_state * -1

            # r_state = (state - min_range) / (max_range - min_range)
            # r_state = 1 - r_state

            r_state = (state - 1) / (max_range - 1)
            r_state = 1 - r_state if r_state > 0 else 0
            r_state = np.power(r_state, 3)

            r_state = np.clip(r_state, a_min=-1.0, a_max=1.0)
        return r_state

    def r_state_lin_norm_Inverted(self, state):
        r_state = 0.0
        if state > self._maxqos:
            r_state = -1.0
        else:
            r_state = 1 - (state - self._minqos) / (self._maxqos - self._minqos)
            r_state = np.clip(r_state, a_min=0.0, a_max=1.0)
        return r_state

    def reward_alpha(self, observation): #  Optimization vars: thpt_glo, cDELAY, cTIMEP, state, mem_use
        thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use = observation.numpy()
        reward = -1.0
        thpt_loss = False

        if thpt_glo >= self._max_thpt:
            self._max_thpt = thpt_glo
            stop = False
            thpt_loss = False
        else:
            measure = self._max_thpt - thpt_glo
            if measure / self._max_thpt > 0.05:  # decrease is greater than 5%
                thpt_loss = True
            else:
                thpt_loss = False

        if (cDELAY > self._window_time and thpt_loss) or (state > self._maxqos) or (state > mem_use > self._minqos):
            reward = -1.0
        elif state >= mem_use:
            if state > self._minqos:
                reward = 1.0
            elif state <= self._minqos:
                reward = -1.0
        elif state >= mem_use:
            reward = -1.0
            
        if mem_use < self._minqos:
            reward = -1.0
        elif mem_use > self._maxqos:
            reward = -1.0

        if cTIMEP == 0.0:
            reward = -1.0
        
        reward = np.round(reward * 10000) / 10000
        reward = np.clip(reward, a_min=-1.0, a_max=1.0)
        return reward

    def reward_beta(self, observation): # Optimizaitn vars: thpt_glo, cDELAY, state
        thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use = observation.numpy()

        max_cDELAY = 20000
        min_cDELAY = 0
        # in GB/s 300 per executor 
        max_thpt_g = 2400
        min_thpt_g = 0

        reward = 0.0

        thpt_normalized = 2 * (thpt_glo - min_thpt_g) / (max_thpt_g - min_thpt_g) -1 # normalized between [-1,1]
        state_normalized = 2 * (self._maxqos - state) / (self._maxqos - self._minqos) -1 # normalized between [-1,1]
        cDELAY_normalized =2 * (max_cDELAY - cDELAY) / (max_cDELAY - min_cDELAY) -1 # normalized between [-1,1]

        reward =  100 * (thpt_normalized + state_normalized + cDELAY_normalized)

        return reward

    def reward_function2(self, observation):
        try:
            thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, qosbase = observation.numpy()
            if any(np.isnan(observation.numpy())) or any(np.isinf(observation.numpy())):
                raise ValueError("Observation contains NaN or inf values")

            reward = 0.0
            delay_penalty = 0.0
            processing_penalty = 0.0
            thpt_reward = 0.0

            self._avg_cDELAY = (self._avg_cDELAY + cDELAY) / 2
            self._avg_cTIMEP = (self._avg_cTIMEP + cTIMEP) / 2

            if self._avg_cDELAY <= self._window_time * 2:
                delay_penalty = 0.0
                if self._avg_cDELAY < self._window_time*0.7:
                    processing_penalty = -2.0
            else:
                # delay_penalty = np.clip((cDELAY - self._window_time)**3 / self._window_time, 0.0, 1.0)
                delay_penalty = np.clip((self._avg_cDELAY - (2 * self._window_time))**2 / 36_000_000, 0.0, 1.0) * -1

            if self._avg_cTIMEP < self._window_time * 1.1:
                processing_penalty = 0.0
                if self._avg_cTIMEP < self._window_time*0.7:
                    processing_penalty = -1.0
            else:
                processing_penalty = np.clip((self._avg_cTIMEP - 1.1 * self._window_time) / 800, 0.0, 1.0) * -1

            if thpt_glo <= self._avg_thpt:
                thpt_reward = 0.0
            else:
                thpt_reward = np.clip((thpt_glo - self._avg_thpt) / self._avg_thpt, 0.0, 1.0)

            self._avg_thpt = (self._avg_thpt + (0.5 * thpt_glo)) / 2.0

            reward = thpt_reward + (delay_penalty + processing_penalty)
            reward = np.round(reward * 1000.0) / 1000.0
            reward = np.clip(reward, a_min=-1.0, a_max=1.0)

            if np.isnan(reward) or np.isinf(reward):
                raise ValueError("Reward calculation resulted in NaN or inf values")

            return reward

        except Exception as e:
            # Handle and log the error
            tf.print("Error in reward function:", e)
            return 0.0



class PPOAgentMQ:
    def __init__(self, start_state, upper_limit, lower_limit):
        self.env = MqEnvironment(upper_limit, lower_limit)
        self.agent = Agent(self.env)
        self.buffer = False
        self.minqos = lower_limit
        self.maxqos = upper_limit

        self._last_state = self.env.reset()
        # print('NO Expand: {}\n'.format(self._last_state))
        time_step = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), self.env.reset())
        # print('With Expand: {}\n'.format(time_step))
        self._last_action = self.agent.getAction(time_step)
        self._last_action = self._last_action

        # self._last_action = self.agent.getAction(self.env.reset())
        # self._last_action = tf.nest.map_structure(lambda x: tf.squeeze(x, axis=[0]),self._last_action)
        self._batch_size = self.agent.batch_size
        self.env._current_action = self._last_action
        self._last_reward = None
        self._first_exec = True

        self._global_step = 0

    def step(self, _new_state):

        new_state = tf.convert_to_tensor(_new_state, dtype=tf.float32)
        last_time_step = self.env.current_time_step()
        current_time_step = self.env.mq_step(self._last_action, new_state, self._global_step)
        self.agent.addToBuffer(last_time_step, self._last_action, current_time_step)

        self.agent.train(self._global_step)
        self._global_step = self._global_step + 1

        # Needs to transform outer dimension because of tf_policy.py's _maybe_reset_state function
        tmp_ts = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), current_time_step)
        tmp_action = self.agent.getAction(tmp_ts)
        self._last_action = tmp_action
      
        action = self._last_action.action.numpy()
        # # Return action -1 because the actions are mapped to 0,1,2 need to -> -1, 0, 1
        # action = self._last_action.action.numpy() - 1
        action = self._last_action.action.numpy()
        if action < 1:
            action = -1

        
        print('Action: {}'.format(action))

        return action

    def finish(self, last_state):
        new_state = tf.convert_to_tensor(last_state, dtype=tf.float32)
        last_time_step = self.env.current_time_step()
        current_time_step = self.env.mq_step(self._last_action, new_state, self._global_step)
        self.agent.addToBuffer(last_time_step, self._last_action, current_time_step)

        return 0



##########################################################################################
# Cython API
##########################################################################################

cdef public object createPPOAgent(float* start_state, int qosmin, int qosmax):
    state = []
    for i in range(8):
        state.append(start_state[i])
    
    # return PPOAgentMQ(state, qosmax, qosmin)
    agent = PPOAgentMQ(state, qosmax, qosmin)
    action = agent.step(state)

    return agent

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
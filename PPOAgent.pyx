#!./venv/bin/python3
# cython: language_level=3

##########################################################################################
# PPO Imports
##########################################################################################
import tensorflow as tf
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.specs import tensor_spec
from tf_agents.specs import array_spec
from tf_agents.specs import BoundedTensorSpec
from tf_agents.specs import TensorSpec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import tf_py_environment

from tf_agents.networks import value_network
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment

from cpython cimport array
import csv

# TESTS
import random

LOGS_DIR =  '/tmp/'
CSV_FILE = LOGS_DIR + "/ppo_logs_01.csv"
LOG_FILE = LOGS_DIR + "/ppo_logs_01.log"

MODEL_NAME = 'PPO-01'

#  Stats settings
AGGREGATE_STATS_EVERY = 20  # steps

GLOBAL_EPSILON = 0.2
GLOBAL_EPOCHS = 10       #3
GLOBAL_GAMMA = 0.99

# PPO Agent 
class PPOClipped:

    def __init__(self, env):
        # Constants
        self.actor_fc_layers = (64,64)
        self.value_fc_layers = (64,64)
        self.epsilon = GLOBAL_EPSILON
        self.epochs = GLOBAL_EPOCHS

        self.time_step_tensor_spec = tensor_spec.from_spec(env.time_step_spec())
        self.observation_tensor_spec = tensor_spec.from_spec(env.observation_spec())
        self.action_tensor_spec = tensor_spec.from_spec(env.action_spec())

        # Building the PPO Agent
        self.actor_net = self.createActorNet()
        self.value_net = self.createValueNet()
        self.optimizer = self.createOptimizer()
        self.train_step_counter = tf.Variable(0)

        self.ppo_agent = self.createPPOAgent()

        self.batch_size = 64
        self.replay_buffer = self.createReplayBuffer()
        self.iterator = self.createBufferIterator()


    def createActorNet(self):
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            input_tensor_spec= self.observation_tensor_spec,
            output_tensor_spec= self.action_tensor_spec,
            fc_layer_params=self.actor_fc_layers,
            # activation_fn=tf.keras.activations.tanh,
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.Orthogonal(seed=self.epochs),
            seed_stream_class=tfp.util.SeedStream
        )
        return actor_net

    def createValueNet(self):
        value_net = value_network.ValueNetwork(
            input_tensor_spec= self.observation_tensor_spec,
            fc_layer_params=self.value_fc_layers,
            # activation_fn=tf.keras.activations.tanh,
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.Orthogonal(seed=self.epochs)
        )
        return value_net

    def createOptimizer(self):
        initial_learning_rate = 1e-3
        decay_steps = 1000
        decay_rate = 0.96
        learning_rate = tf.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps, decay_rate, staircase=True
        )
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        learning_rate = 2.5e-4
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        return optimizer


    def createPPOAgent(self):
        agent_ppo = ppo_clip_agent.PPOClipAgent(
            time_step_spec=self.time_step_tensor_spec,
            action_spec=self.action_tensor_spec,
            optimizer=self.optimizer,
            normalize_observations=True,
            # normalize_observations=False,
            normalize_rewards=True,
            # normalize_rewards=False,
            actor_net=self.actor_net,
            value_net=self.value_net,
            importance_ratio_clipping=self.epsilon,
            num_epochs=self.epochs,
            use_gae=True,
            train_step_counter=self.train_step_counter,
        )
        agent_ppo.initialize()
        agent_ppo.train_step_counter.assign(0)
        # (Optional) Optimize by wrapping some of this code in a graph using TF function.
        agent_ppo.train = common.function(agent_ppo.train)
        return agent_ppo

    def createReplayBuffer(self):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec= self.ppo_agent.collect_policy.trajectory_spec,
            # self.ppo_agent.policy.trajectory_spec,
            batch_size=1,
            max_length=10000)
        return replay_buffer

    def addToBuffer(self, traj):
        self.replay_buffer.add_batch(traj)
        
    def createBufferIterator(self):
        n_step_update = 10
        batch_size = self.batch_size
        dataset = self.replay_buffer.as_dataset(
            num_steps=n_step_update,
            num_parallel_calls=3,
            sample_batch_size=batch_size
            ).prefetch(batch_size)
        iterator = iter(dataset)
        return iterator

    def train(self):
        experience, unused_info = next(self.iterator)
        with open(LOG_FILE, mode='a+') as logFile:
            logFile.write('{}\n'.format(experience))
        self.ppo_agent.train(experience)
        # print('Step Counter: {0}'.format(self.train_step_counter))

    def getAction(self, time_step):
        # return self.ppo_agent.policy.action(time_step)
        return self.ppo_agent.collect_policy.action(time_step)



class MqEnvironment(py_environment.PyEnvironment):
    
    def __init__(self, maxqos, minqos):
        self._observation_spec = TensorSpec(shape=(8,), dtype=tf.float32, name='observation')
        self._action_spec = BoundedTensorSpec(
            shape=(), dtype=tf.int32, minimum=-1, maximum=1, name='action')
        self._reward_spec = TensorSpec(shape=(), dtype=tf.float32, name='reward')

        self._maxqos = maxqos
        self._minqos = minqos

        self._rewards = 0
        self._current_time_step = None
        self._action = None
        self._discount = GLOBAL_GAMMA
        with open(CSV_FILE, mode='w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['thpt_glo', 'thpt_var', 'cDELAY', 'cTIMEP', 'RecSparkTotal', 'RecMQTotal', 'state', 'mem_use','r_thpt_glo', 'r_thpt_var', 'r_cDELAY', 'r_cTIMEP', 'r_RecSparkTotal', 'r_RecMQTotal', 'r_state', 'r_mem_use','r_norm'])

        self._max_thpt = 100
        self._max_cDELAY = 20000
        self._max_cTIMEP = 10000

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def reward_spec(self):
        return self._reward_spec
    
    def _reset(self):
        observation_zeros = tf.zeros((8,), dtype=tf.float32)
        reward = tf.convert_to_tensor(1, dtype=tf.float32)
        self._current_time_step = ts.transition(observation_zeros, reward=reward, discount=1.0)
        
        return self._current_time_step
    
    def _step(self, action):
        self._action = action
        return self._current_time_step

    def mq_step(self, action, state):
        observation = tf.convert_to_tensor(state, dtype=tf.float32)
        reward = self.get_reward(observation)
        print("R: {0}".format(reward))
        self._current_time_step = ts.transition(observation, reward=reward, discount=self._discount)
        self._action = action
        return self._current_time_step
    
    def get_reward(self, observation):
    # [total_throughput, thpt_variation, proc_t, sche_t, msgs_to_spark, msgs_in_gb, ready_mem, spark_thresh]
        thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use = observation.numpy()
        lst_thpt_glo, lst_thpt_var, lst_cDELAY, lst_cTIMEP, lst_RecSparkTotal, lst_RecMQTotal, lst_state, lst_mem_use = self.current_time_step().observation.numpy()
        r_thpt_glo, r_thpt_var, r_cDELAY, r_cTIMEP, r_RecSparkTotal, r_RecMQTotal, r_state, r_mem_use = np.zeros(8, dtype=np.float32)
                
        # MEMORY USED
        if mem_use > self._maxqos:
            r_mem_use = 0.0
        else:
            r_mem_use = (np.exp(5*mem_use) - self._maxqos) / (self._maxqos - self._minqos)
            r_mem_use = np.clip(r_mem_use, a_min=0.0, a_max=1.0)

        # STATE
        if state > self._maxqos:
            r_state = 0.0
        else:
            r_state = (np.exp(5*r_state) - self._maxqos) / (self._maxqos - self._minqos)
            r_state = np.clip(r_state, a_min=0.0, a_max=1.0)
        
        # THPT_GLO
        self._max_thpt = max(self._max_thpt, thpt_glo)
        r_thpt_glo = (np.exp(5*thpt_glo)) / (self._max_thpt)
        r_thpt_glo = np.clip(r_thpt_glo, a_min=0.0, a_max=1.0)

        # PROCESSING DELAY
        self._max_cTIMEP = max(10000, cTIMEP)
        r_cTIMEP = 1 - (cTIMEP / self._max_cTIMEP)
        r_cTIMEP = np.exp(5 * cTIMEP)
        r_cDELAY = np.clip(r_cTIMEP, a_min=0.0, a_max=1.0)

        # SCHEDULING DELAY
        self._max_cDELAY = max(20000, cDELAY)
        r_cDELAY_tmp = 1 - (cDELAY / self._max_cDELAY)
        r_cDELAY_tmp = np.exp(5 * r_cDELAY_tmp)
        r_cDELAY = np.clip(r_cDELAY_tmp, a_min=0.0, a_max=1.0)


        # rewards = np.array([r_thpt_glo, r_thpt_var, r_cDELAY, r_cTIMEP, r_RecSparkTotal, r_RecMQTotal, r_state, r_mem_use])
        # normalize_rewards = rewards / np.sum(rewards)

        rewards_p = np.array([r_thpt_glo, r_cDELAY, r_cTIMEP, r_state]) # Removed mem_use
        reward = np.prod(rewards_p) / np.sum(rewards_p) # Normalized Array
        
        reward = np.clip(reward, a_min=0.0, a_max=1.0)

        with open(CSV_FILE, mode='a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use,r_thpt_glo, r_thpt_var, r_cDELAY, r_cTIMEP, r_RecSparkTotal, r_RecMQTotal, r_state, r_mem_use, reward])
        return tf.convert_to_tensor(reward, dtype=tf.float32)


class PPOAgentMQ:
    def __init__(self, start_state, upper_limit, lower_limit):
        self.env = MqEnvironment(upper_limit, lower_limit)
        self.ppo_agent = PPOClipped(self.env)
        self.buffer = False

        self._last_state = self.env.reset()
        self._last_action = self.ppo_agent.getAction(self._last_state)
        self._batch_size = self.ppo_agent.batch_size
        self.env._current_action = self._last_action
        self._last_reward = None
        self._first_exec = True
        # print("I'm PPO!")
        # with open(LOG_FILE, mode='a+') as logFile:
        #     logFile.write('{}, {}, {}\n'.format('LAST_STATE', 'LAST_TIMESTEP', 'LAST_ACTION'))
        
    def step(self, _new_state):
      
        new_state = tf.convert_to_tensor(_new_state, dtype=tf.float32)
        last_time_step = self.env.current_time_step()
        last_action = self._last_action
        current_time_step = self.env.mq_step(last_action, new_state)
        
        traj = trajectory.from_transition(last_time_step, last_action, current_time_step)
        traj_batched = tf.nest.map_structure(lambda t: tf.stack([t] * 1), traj)
        self.ppo_agent.addToBuffer(traj_batched)
        
        # if self.ppo_agent.replay_buffer.num_frames().numpy() > self._batch_size:
        # if self.ppo_agent.replay_buffer.num_frames().numpy() > (self._batch_size * 10):
        # if not (self.ppo_agent.replay_buffer.num_frames().numpy() % (self._batch_size/2)):
        if not (self.ppo_agent.replay_buffer.num_frames().numpy() % self._batch_size):
            self.ppo_agent.train()
            
        self._last_action = self.ppo_agent.getAction(current_time_step)
        # with open(LOG_FILE, mode='a+') as logFile:
        #     logFile.write('{}, {}, {}\n'.format(last_time_step.observation.numpy(), last_action.action.numpy(), current_time_step.observation.numpy()))


        # if self.ppo_agent.ppo_agent.train_step_counter < 300:
        #     action = random.randint(0,1)
        # else:
        # # Return action -1 because the actions are mapped to 0,1,2 need to -> 1, 0, 1
        #     action = self._last_action.action.numpy() - 1
        action = self._last_action.action.numpy() - 1
        
        return action
        #return 1

    def finish(self, last_state):
        new_state = tf.convert_to_tensor(last_state, dtype=tf.float32)
        last_time_step = self.env.current_time_step()
        last_action = self._last_action
        current_time_step = self.env.mq_step(last_action, last_state)

        # with open(LOG_FILE, mode='a+') as logFile:
        #     logFile.write('{}, {}, {}\n'.format(last_time_step, last_action, current_time_step))

        traj = trajectory.from_transition(last_time_step, last_action, current_time_step)
        traj_batched = tf.nest.map_structure(lambda t: tf.stack([t] * 1), traj)
        self.ppo_agent.addToBuffer(traj_batched)

        return 0



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
    # action = random.randint(0,1)
    # action = random.choices([-1, 0, 1], [0.1, 0.7, 0.2])[0]
    # action = 0

    #return random.randint(0, 1)
    print("A: {0}".format(action))
    return action

cdef public void finish(object agent, float* last_state):
    state = []
    for i in range(8):
        state.append(last_state[i])
    
    agent.finish(state)
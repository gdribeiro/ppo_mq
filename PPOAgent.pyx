#!./venv/bin/python3
# cython: language_level=3


##########################################################################################
# PPO Imports
##########################################################################################
import tensorflow as tf
import numpy as np

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
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment

from cpython cimport array
import csv


LOGS_DIR =  '/tmp/'
CSV_FILE = LOGS_DIR + "/ppo_logs_01.csv"
LOG_FILE = LOGS_DIR + "/ppo_logs_01.log"

MODEL_NAME = 'PPO-01'

#  Stats settings
AGGREGATE_STATS_EVERY = 20  # steps

GLOBAL_EPSILON = 0.2
GLOBAL_EPOCHS = 3
GLOBAL_GAMMA = 0.99

# PPO Agent 
class PPOClipped:

    def __init__(self, env):
        # Constants
        # self.actor_fc_layers = (64,64)
        # self.value_fc_layers = (64,64)
        self.actor_fc_layers = (128,128,64)
        self.value_fc_layers = (128,128,64)
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

        with open(LOG_FILE, mode='a+') as logFile:
            logFile.write('Training Data Spec: {}\nBuffer Data Spec: {}\n'.format(self.ppo_agent.training_data_spec, self.replay_buffer.data_spec))


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
            normalize_rewards=True,
            actor_net=self.actor_net,
            value_net=self.value_net,
            importance_ratio_clipping=self.epsilon,
            num_epochs=self.epochs,
            use_gae=True,
            train_step_counter=self.train_step_counter,
        )
        agent_ppo.initialize()
        agent_ppo.train_step_counter.assign(0)
        agent_ppo.train = common.function(agent_ppo.train)
        return agent_ppo

    def createReplayBuffer(self):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec= self.ppo_agent.collect_policy.trajectory_spec,
            batch_size= 1,
            max_length=10000)
        return replay_buffer

    def addToBuffer(self, traj):
        self.replay_buffer.add_batch(traj)
        
    def createBufferIterator(self):
        n_step_update = 10
        dataset = self.replay_buffer.as_dataset(
            num_steps=n_step_update,
            sample_batch_size=self.batch_size,
            num_parallel_calls=3, 
            single_deterministic_pass=False
        ).prefetch(self.batch_size)
        iterator = iter(dataset)
        return iterator

    def train(self):
        experience, unused_info = next(self.iterator)
        
        with open(LOG_FILE, mode='a+') as logFile:
            logFile.write('XP: {}\n'.format(experience))

        self.ppo_agent.train(experience)


    def getAction(self, time_step):
        # return self.ppo_agent.policy.action(time_step)
        return self.ppo_agent.collect_policy.action(time_step)



class MqEnvironment(py_environment.PyEnvironment):
    
    def __init__(self, maxqos, minqos):
        self._observation_spec = TensorSpec(shape=(8,), dtype=tf.float32, name='observation')
        self._action_spec = BoundedTensorSpec(
            shape=(), dtype=tf.int32, minimum=-1, maximum=1, name='action')
        self._reward_spec = TensorSpec(shape=(8,), dtype=tf.float32, name='reward')
        self._discount_spec = BoundedTensorSpec(
            shape=(8,), dtype=np.float32, minimum=0., maximum=1., name='discount')

        self._maxqos = maxqos
        self._minqos = minqos

        self._rewards = 0
        self._current_time_step = None
        self._action = None
        self._discount = tf.fill((8,), GLOBAL_GAMMA, dtype=tf.float32)
        with open(CSV_FILE, mode='w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['thpt_glo', 'thpt_var', 'cDELAY', 'cTIMEP', 'RecSparkTotal', 'RecMQTotal', 'state', 'mem_use','r_thpt_glo', 'r_thpt_var', 'r_cDELAY', 'r_cTIMEP', 'r_RecSparkTotal', 'r_RecMQTotal', 'r_state', 'r_mem_use'])


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def reward_spec(self):
        return self._reward_spec
    
    def _reset(self):
        observation_zeros = tf.zeros((8,), dtype=tf.float32)
        reward = tf.zeros((8,), dtype=tf.float32)
        # self._current_time_step = ts.transition(observation_zeros, reward=reward, discount=self._discount.numpy(), outer_dims=())
        self._current_time_step = ts.transition(observation_zeros, reward=reward, discount=self._discount.numpy())
        
        return self._current_time_step
    
    def _step(self, action):
        self._action = action
        return self._current_time_step

    def mq_step(self, action, state):
        observation = tf.convert_to_tensor(state, dtype=tf.float32)
        reward = self.get_reward(observation)
        print("R: {0}".format(reward))
        # self._current_time_step = ts.transition(observation, reward=reward, discount=self._discount.numpy(), outer_dims=())
        self._current_time_step = ts.transition(observation, reward=reward, discount=self._discount.numpy())
        self._action = action
        return self._current_time_step
    
    def get_reward(self, observation):
    # [total_throughput, thpt_variation, proc_t, sche_t, msgs_to_spark, msgs_in_gb, ready_mem, spark_thresh]
        thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use = observation.numpy()
        lst_thpt_glo, lst_thpt_var, lst_cDELAY, lst_cTIMEP, lst_RecSparkTotal, lst_RecMQTotal, lst_state, lst_mem_use = self.current_time_step().observation.numpy()
        r_thpt_glo, r_thpt_var, r_cDELAY, r_cTIMEP, r_RecSparkTotal, r_RecMQTotal, r_state, r_mem_use = np.zeros(8, dtype=np.float32)
                
        max_reward_value = np.finfo(np.float32).max
        # r_mem_use = 2 * (self._minqos-mem_use) / (self._maxqos-self._minqos) +1
        # r_mem_use = 1/mem_use
        # r_mem_use = min(1/mem_use, max_reward_value)
        # Calculate the midpoint between self._minqos and self._maxqos
        # mid_point = (self._minqos + self._maxqos) / 2.0

        # Calculate the distance of mem_use from the midpoint
        # distance_to_mid_point = abs(mem_use - mid_point)

        # Calculate the maximum possible distance from the midpoint,
        # which would be when mem_use is either at self._minqos or self._maxqos
        # max_distance = max(self._maxqos - mid_point, mid_point - self._minqos)

        # Normalize the distance to the range [0, 1]
        # normalized_distance = distance_to_mid_point / max_distance

        # Set the reward to decrease as the distance increases. 
        # It will be 1 when mem_use is at the midpoint, and decrease linearly from there.
        # r_mem_use = 1 - normalized_distance

        # Finally, scale the reward to not exceed max_reward_value
        # r_mem_use = min(r_mem_use, max_reward_value)
        r_mem_use = 2 * (self.maxqos - mem_use) / (self.maxqos - self.minqos) -1 

     
        # r_thpt_glo = thpt_glo - lst_thpt_glo
        # r_thpt_glo = thpt_glo
        # r_thpt_glo = min(thpt_glo, max_reward_value)
        r_thpt_glo = 2 * (thpt_glo - 0) / (2400 - 0) -1 

        # r_thpt_var = thpt_var

    
        # r_cDELAY =  100 * (1-(cDELAY - 10) / (25000))
        # r_cDELAY =  1/r_cDELAY
        # r_cDELAY =  min(1/cDELAY, max_reward_value)
        r_cDELAY =2 * ((20000 - cDELAY) / (20000)) -1
                
        # r_cTIMEP = 100 * (1-(cTIMEP - 10) / (25000))
        # r_cTIMEP = 1/cTIMEP
        # r_cTIMEP = min(1/cTIMEP, max_reward_value)

        # r_state = 100 * (1-(state - 1) / (32))
        # r_state = 1/state
        # r_state = min(1/state, max_reward_value)


        with open(CSV_FILE, mode='a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use,r_thpt_glo, r_thpt_var, r_cDELAY, r_cTIMEP, r_RecSparkTotal, r_RecMQTotal, r_state, r_mem_use])

        rewards = np.array([r_thpt_glo, r_thpt_var, r_cDELAY, r_cTIMEP, r_RecSparkTotal, r_RecMQTotal, r_state, r_mem_use])
       
        return tf.convert_to_tensor(rewards, dtype=tf.float32)


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
        print("I'm PPO!")
        # with open(LOG_FILE, mode='a+') as logFile:
        #     logFile.write('{}, {}, {}\n'.format('LAST_STATE', 'LAST_TIMESTEP', 'LAST_ACTION'))

        
    def step(self, _new_state):
        new_state = tf.convert_to_tensor(_new_state, dtype=tf.float32)
        last_time_step = self.env.current_time_step()
        last_action = self._last_action
        current_time_step = self.env.mq_step(last_action, new_state)
        traj = trajectory.from_transition(last_time_step, last_action, current_time_step)
        traj_batched = tf.nest.map_structure(lambda t: tf.stack([t] * 1), traj)
        traj = traj_batched       
        self.ppo_agent.addToBuffer(traj)
        
        if self.ppo_agent.replay_buffer.num_frames().numpy() > self._batch_size:
            self.ppo_agent.train()

        self._last_action = self.ppo_agent.getAction(current_time_step)
        # with open(LOG_FILE, mode='a+') as logFile:
        #     logFile.write('{}, {}, {}\n'.format(last_time_step.observation.numpy(), last_action.action.numpy(), current_time_step.observation.numpy()))

        action = self._last_action.action.numpy() - 1        
        return action


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

    print("A: {0}".format(action))
    return action

cdef public void finish(object agent, float* last_state):
    state = []
    for i in range(8):
        state.append(last_state[i])
    
    agent.finish(state)
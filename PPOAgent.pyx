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

from tf_agents import networks
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment

from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
import timeit

from cpython cimport array
import csv

# TESTS
import random



LOGS_DIR =  '/tmp/'
CSV_FILE = LOGS_DIR + "/log_rewards.csv"
LOG_FILE = LOGS_DIR + "/log_general.log"
AGENT_FILE = LOGS_DIR + "/log_agent.csv"

# tf_log_metrics_dir = LOGS_DIR + "/tf-metrics"
# summary_writer = tf.summary.create_file_writer(tf_log_metrics_dir)


MODEL_NAME = 'SparkPPO'

GLOBAL_BUFFER_SIZE = 100
GLOBAL_EPSILON = 0.1    # 0.1 is much more stable!
GLOBAL_EPOCHS = 3       #3 10 - 3 is BEST
GLOBAL_GAMMA = 0.99
GLOBAL_BATCH = 2   # 2 small is better
GLOBAL_STEPS = 2 # 2 is BEST


# PPO Agent 
class PPOClipped:

    def __init__(self, env):
        self.policy_fc_layers= (32,32,32,32,32)
        self.actor_fc_layers = self.policy_fc_layers
        self.value_fc_layers = self.policy_fc_layers
        self.epsilon = GLOBAL_EPSILON
        self.gamma = GLOBAL_GAMMA
        self.epochs = GLOBAL_EPOCHS

        self.time_step_tensor_spec = tensor_spec.from_spec(env.time_step_spec())
        self.observation_tensor_spec = tensor_spec.from_spec(env.observation_spec())
        self.action_tensor_spec = tensor_spec.from_spec(env.action_spec())

        self.actor_net = self.createActorNet()
        self.value_net = self.createValueNet()
        self.optimizer = self.createOptimizer()
        self.train_step_counter = tf.Variable(0)

        self.ppo_agent = self.createPPOAgent()

        self.batch_size = GLOBAL_BATCH
        self.num_steps = GLOBAL_STEPS
        self.buffer_size = GLOBAL_BUFFER_SIZE
        self.replay_buffer = self.createReplayBuffer()
        self.iterator = self.createBufferIterator()

        self._loss = 10000
        self._eval = False
        self._counter = 0

        with open(AGENT_FILE, mode='w') as agentLog:
            writer = csv.writer(agentLog)
            writer.writerow(['step', 'StepCounter', 'Loss'])

    def createActorNet(self):
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            input_tensor_spec= self.observation_tensor_spec,
            output_tensor_spec= self.action_tensor_spec,
            input_fc_layer_params= None,
            output_fc_layer_params= self.policy_fc_layers,
            lstm_size=(8,)
        )
        return actor_net

    def createValueNet(self):
        value_net = value_rnn_network.ValueRnnNetwork(
            input_tensor_spec= self.observation_tensor_spec,
            input_fc_layer_params= None,
            output_fc_layer_params= self.policy_fc_layers,
            lstm_size=(8,)
        )
        return value_net

    def createOptimizer(self):
        learning_rate = 3e-4
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        return optimizer


    def createPPOAgent(self):
        agent_ppo = ppo_clip_agent.PPOClipAgent(
            time_step_spec=self.time_step_tensor_spec,
            action_spec=self.action_tensor_spec,
            actor_net=self.actor_net,
            value_net=self.value_net,
            optimizer=self.optimizer,
            # normalize_observations=False,
            # normalize_rewards=False,
            use_td_lambda_return=True,
            importance_ratio_clipping=self.epsilon,
            value_clipping = 0.1,
            num_epochs=self.epochs,
            use_gae=True,
            train_step_counter=self.train_step_counter,
            # greedy_eval=False,
            greedy_eval=True,
            entropy_regularization=0.01,
            value_pred_loss_coef=1.0,
            policy_l2_reg = 0.0001,
            value_function_l2_reg = 0.0001,
            name=MODEL_NAME
        )
        

        agent_ppo.initialize()
        print('ActorDistributionNetwork: {}\n'.format(agent_ppo.actor_net.summary()))
        print('ValueRnnNetwork: {}\n'.format(agent_ppo._value_net.summary()))

        agent_ppo.train_step_counter.assign(0)
        # (Optional) Optimize by wrapping some of this code in a graph using TF function.
        # agent_ppo.train = common.function(agent_ppo.train)
        agent_ppo.train = common.function(agent_ppo.train, autograph=False)
        return agent_ppo

    def createReplayBuffer(self):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec= self.ppo_agent.collect_policy.trajectory_spec,
            batch_size=1,
            max_length=self.buffer_size)
        return replay_buffer

    def addToBuffer(self, last_time_step, last_action, current_time_step):
        traj = trajectory.from_transition(last_time_step, last_action, current_time_step)
        traj_batched = tf.nest.map_structure(lambda t: tf.stack([t] * 1), traj)
        self.replay_buffer.add_batch(traj_batched)

    def createBufferIterator(self):
        dataset = self.replay_buffer.as_dataset(
            num_steps= self.num_steps,
            num_parallel_calls=3,
            sample_batch_size= self.batch_size
        ).prefetch(self.batch_size)
        iterator = iter(dataset)
        return iterator

    def train(self, global_step):
        self._eval = True if self._loss < -10 else False

        if not self._eval:
            if not (self.replay_buffer.num_frames().numpy() % (self.num_steps * self.batch_size)):
                experience, unused_info = next(self.iterator)
                self._loss, _ = self.ppo_agent.train(experience)

                with open(AGENT_FILE, mode='a+', newline='') as agentLog:
                    writer = csv.writer(agentLog)
                    writer.writerow([global_step, self.train_step_counter.numpy(), self._loss.numpy()])
                print('TrainStep: {},\t LOSS: {}\n'.format(self.train_step_counter.numpy(), self._loss.numpy()))

                # Because ON-POLICY training
                self.replay_buffer.clear()
        else:
            self._counter += 1
            self._eval = False if self._counter > 600 else True

    def getAction(self, time_step):
        if not self._eval:
            collect_policy_state = self.ppo_agent.collect_policy.get_initial_state(batch_size=1)
            collect_action = self.ppo_agent.collect_policy.action(time_step, collect_policy_state)
            print('**TRAIN**: collect_action: {}'.format(collect_action.action.numpy()))
            action = collect_action
        else:
            policy_state = self.ppo_agent.collect_policy.get_initial_state(batch_size=1)
            action = self.ppo_agent.collect_policy.action(time_step, policy_state)
            print('**EVAL**: Action: {}'.format(action.action.numpy()))

        return action
    

class MqEnvironment(py_environment.PyEnvironment):

    def __init__(self, maxqos, minqos):
        self._observation_spec = TensorSpec(shape=(8,), dtype=tf.float32, name='observation')
        # self._action_spec = BoundedTensorSpec(shape=(), dtype=tf.int32, minimum=minqos, maximum=maxqos, name='action')
        # self._action_spec = BoundedTensorSpec(shape=(), dtype=tf.int32, minimum=0, maximum=2, name='action')
        self._action_spec = BoundedTensorSpec(shape=(), dtype=tf.int32, minimum=0, maximum=1, name='action')
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

        # self._max_cDELAY = 10000
        self._max_cDELAY = 2000
        self._max_cTIMEP = 2000
        self._avg_thpt = 0
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

        # reward = self.reward_gamma(observation)
        # reward = self.reward_beta(observation)
        reward = self.reward_alpha(observation)
        
        self._rewards += reward
        print('** Reward: {}\n** Total Rewards: {}'.format(reward, self._rewards))

        with open(CSV_FILE, mode='a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([global_step, thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use, reward])
        return tf.convert_to_tensor(reward, dtype=tf.float32)

    def reward_gamma(self, observation): # Opotimization vars: thpt_glo,  cDELAY, cTIMEP, state
        thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use = observation.numpy()
        lst_thpt_glo, lst_thpt_var, lst_cDELAY, lst_cTIMEP, lst_RecSparkTotal, lst_RecMQTotal, lst_state, lst_mem_use = self.current_time_step().observation.numpy()

        reward = 0.0

        if cDELAY > self._window_time or cTIMEP > self._window_time:
            reward = self.r_state_lin_norm_cost(state)
            if cTIMEP > self._window_time * 1.25:
                reward = -1.0
            elif cDELAY > self._window_time * 5:
                reward = -1.0
            
            if cDELAY < lst_cDELAY:
                # reward = reward + 0.001
                reward = reward + 0.01

        elif thpt_glo > self._avg_thpt:
            r_cDELAY = self.r_cDELAY_lin_norm_Inverted_original(cDELAY)
            r_cTIMEP = self.r_cTIMEP_lin_norm_Inverted_original(cTIMEP)
            r_state  = self.r_state_lin_norm_Inverted(state)
            rewards_p = np.array([r_cDELAY, r_cTIMEP, r_state], dtype=np.float32)
            weights = np.array([1, 1, 1])
            reward = np.average(rewards_p, weights=weights)
        else:
            reward = 0.0001

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
        if state > self._maxqos:
            r_state = -1.0
        else:
            r_state = (state) / (self._maxqos/2)
            r_state = r_state * -1
            r_state = np.clip(r_state, a_min=-1.0, a_max=0.0)
        return r_state

    def r_state_lin_norm_Inverted(self, state):
        r_state = 0.0
        if state > self._maxqos:
            r_state = 0.0
        else:
            r_state = 1 - (state - self._minqos) / (self._maxqos - self._minqos)
            r_state = np.clip(r_state, a_min=0.0, a_max=1.0)
        return r_state

    def reward_alpha(self, observation): #  Optimization vars: thpt_glo, cDELAY, cTIMEP, state, mem_use
        thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use = observation.numpy()
        reward = -1.0
        thpt_loss = False
        stop = False

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
        elif not stop and state >= mem_use:
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


class PPOAgentMQ:
    def __init__(self, start_state, upper_limit, lower_limit):
        self.env = MqEnvironment(upper_limit, lower_limit)
        self.agent = PPOClipped(self.env)
        self.buffer = False
        self.minqos = lower_limit
        self.maxqos = upper_limit

        self._last_state = self.env.reset()
        time_step = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), self.env.reset())
        self._last_action = self.agent.getAction(time_step)
        self._last_action = tf.nest.map_structure(lambda x: tf.squeeze(x, axis=[0]),self._last_action)
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
        self._last_action = tf.nest.map_structure(lambda x: tf.squeeze(x, axis=[0]), tmp_action)

        # # Return action -1 because the actions are mapped to 0,1,2 need to -> -1, 0, 1
        # action = self._last_action.action.numpy() + 1
        # action = self._last_action.action.numpy() + self.minqos
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
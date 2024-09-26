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

from tf_agents import networks
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment

from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics

from cpython cimport array
import csv

# TESTS
import random


LOGS_DIR =  '/tmp/'
CSV_FILE = LOGS_DIR + "/log_rewards.csv"
LOG_FILE = LOGS_DIR + "/log_general.log"
AGENT_FILE = LOGS_DIR + "/log_agent.csv"

tf_log_metrics_dir = LOGS_DIR + "/tf-metrics"
summary_writer = tf.summary.create_file_writer(tf_log_metrics_dir)


MODEL_NAME = 'PPO-01'


GLOBAL_EPSILON = 0.1
GLOBAL_EPOCHS = 10       #3
GLOBAL_GAMMA = 0.99
GLOBAL_BATCH = 8 # 4
GLOBAL_STEPS = 16 # 32


# PPO Agent 
class PPOClipped:

    def __init__(self, env):
        self.policy_fc_layers= (16,16,16,16)
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
        self.replay_buffer = self.createReplayBuffer()
        self.iterator = self.createBufferIterator()

        self._loss = 10000
        self._eval_rewards = 0
        self._eval = False
        self._counter = 0

        with open(AGENT_FILE, mode='w') as agentLog:
            writer = csv.writer(agentLog)
            writer.writerow(['StepCounter', 'Loss'])


    def createActorNet(self):
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            input_tensor_spec= self.observation_tensor_spec,
            output_tensor_spec= self.action_tensor_spec,
            fc_layer_params=self.actor_fc_layers,
            activation_fn=tf.keras.activations.tanh
            # activation_fn=tf.keras.activations.relu,
        )
        return actor_net

    def createValueNet(self):
        value_net = value_network.ValueNetwork(
            input_tensor_spec= self.observation_tensor_spec,
            fc_layer_params=self.value_fc_layers,
            # activation_fn=tf.keras.activations.relu,
            activation_fn=tf.keras.activations.tanh
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
            normalize_observations=False,
            normalize_rewards=False,
            use_td_lambda_return=True,
            importance_ratio_clipping=self.epsilon,
            # discount_factor=0.95,
            num_epochs=self.epochs,
            use_gae=True,
            train_step_counter=self.train_step_counter,
            # greedy_eval=False,
            greedy_eval=True,
            entropy_regularization=0.02,
            value_pred_loss_coef=1.0,
            policy_l2_reg = 0.01,
            value_function_l2_reg = 0.01,
            shared_vars_l2_reg = 0.0
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
            # data_spec= self.ppo_agent.policy.trajectory_spec,
            batch_size=1,
            max_length=1000)

        return replay_buffer

    def addToBuffer(self, last_time_step, last_action, current_time_step):
        if not self._eval:
            traj = trajectory.from_transition(last_time_step, last_action, current_time_step)
            traj_batched = tf.nest.map_structure(lambda t: tf.stack([t] * 1), traj)
            self.replay_buffer.add_batch(traj_batched)
        
    def createBufferIterator(self):
        n_step_update = GLOBAL_STEPS
        batch_size = self.batch_size
        dataset = self.replay_buffer.as_dataset(
            num_steps=n_step_update,
            num_parallel_calls=3,
            sample_batch_size=batch_size
            ).prefetch(batch_size)
        iterator = iter(dataset)
        return iterator

    def train(self):
        self._eval = True if self._loss < -0 else False

        if not self._eval:
            # if not (self.replay_buffer.num_frames().numpy() % GLOBAL_STEPS):
            if not (self.replay_buffer.num_frames().numpy() % (self.num_steps * self.batch_size)):
                experience, unused_info = next(self.iterator)
                self._loss, _ = self.ppo_agent.train(experience)

                self.replay_buffer.clear()

                with open(AGENT_FILE, mode='a+', newline='') as agentLog:
                    writer = csv.writer(agentLog)
                    writer.writerow([self.train_step_counter.numpy(), self._loss.numpy()])
                print('TrainStep: {},\t LOSS: {}\n'.format(self.train_step_counter.numpy(), self._loss.numpy()))
        else:
            self._counter += 1
            if self._counter > 2000:
                self._eval = False


        
    def getAction(self, time_step):
        collect_policy_state = self.ppo_agent.collect_policy.get_initial_state(self.batch_size)
        collect_action = self.ppo_agent.collect_policy.action(time_step, collect_policy_state)

        policy_state = self.ppo_agent.policy.get_initial_state(self.batch_size)
        action = self.ppo_agent.policy.action(time_step, policy_state)

        e = True if collect_action.action.numpy() == action.action.numpy() else False
        print('Action: {}'.format(action.action.numpy()))
        print('Collect Action: {}'.format(collect_action.action.numpy()))
        print('Equals: {}'.format(e))
        

        returned_action = action if self._eval else collect_action

        return returned_action
    

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

        self._max_thpt = 100
        self._max_cDELAY = 10000
        self._max_cTIMEP = 2000

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
                
        # THPT_VAR
        r_thpt_var= self.r_thpt_var_lin_norm(thpt_var)
        # THPT_GLO
        r_thpt_glo  = self.r_thpt_glo_lin_norm(thpt_glo)
        # cDELAY: Scheculing Delay
        r_cDELAY    = self.r_cDELAY_lin_norm_Inverted_original(cDELAY)
        # cTIMEP: Processing Delay
        r_cTIMEP    = self.r_cTIMEP_lin_norm_Inverted(r_cTIMEP)
        # STATE
        r_state     = self.r_state_lin_norm_Inverted(state)
        # MEMORY USED
        r_mem_use   = self.r_mem_use_lin_norm_Inverted(mem_use)
        # r_RecSparkTotal
        r_RecSparkTotal = self.r_RecSparkTotal_reward(RecSparkTotal, lst_RecSparkTotal)
        # r_RecMQTotal
        r_RecMQTotal =  self.r_RecMQTotal_reward(RecMQTotal, lst_RecMQTotal)


        # rewards_p = np.array([r_thpt_glo, r_cDELAY, r_cTIMEP, r_state, r_mem_use], dtype=np.float32)
        rewards_p = np.array([r_thpt_glo, r_cDELAY, r_cTIMEP, r_state], dtype=np.float32)
        # rewards_p = np.array([r_thpt_glo ,r_thpt_var, r_cDELAY, r_cTIMEP, r_state, r_mem_use], dtype=np.float32)
        # rewards_p = np.array([r_thpt_glo ,r_thpt_var, r_cDELAY, r_cTIMEP, r_RecSparkTotal, r_RecMQTotal, r_state, r_mem_use], dtype=np.float32)
        # rewards_p = np.array([r_thpt_glo, r_cDELAY, r_cTIMEP, r_state], dtype=np.float32)
        # weights = np.array([1, 15, 1, 7, 1]) # Good np.array([3, 15, 1, 10, 1])
        weights = np.array([1, 4, 1, 1]) # Good np.array([3, 15, 1, 10, 1])
        # weights = np.array([1, 1, 3, 1, 1, 2]) # Good np.array([3, 15, 1, 10, 1])
        # weights = np.array([0.1, 2, 10, 0.5, 0.5, 0.5, 4, 1]) # Good np.array([3, 15, 1, 10, 1])
        # weights = np.array([35, 100, 10, 50])
        reward = np.average(rewards_p, weights=weights)
        reward = np.round(reward * 100) / 100
        reward = np.clip(reward, a_min=0.0, a_max=0.5)
        
        window_time = 2000.0

        # if cDELAY < (window_time / 2):
        #     reward = 0.5
        # elif cDELAY <= window_time:
        #     reward = 1.0
        # elif cDELAY <= (window_time * 2):
        #     reward = 0.7
        # elif cDELAY <= (window_time * 4):
        #     reward = 0.3
        # elif (lst_state - state) > 0:
        #     reward = (lst_state - state)/10000
        # else:
        #     reward = 0.0

        # if (lst_state - state) > 0:
        #     if reward >= 0.5:
        #         reward += 0.2
        #     elif reward <= 0.3:
        #         reward += 0.1
        #     else:
        #         reward = 0.1

        # if lst_thpt_var >= thpt_var:
        # if lst_cDELAY >= cDELAY:
        #     if cDELAY <= window_time:
        #         reward = 1.0
        #     elif cDELAY <= (window_time * 2):
        #         reward = 0.5
        #     elif cDELAY <= (window_time * 4):
        #         reward = 0.05
        #     else:
        #         reward = 0.01
        # else:
        #     reward = 0.0

        # if cDELAY <= window_time and lst_thpt_glo >= thpt_glo and state < self._maxqos:
        if cDELAY <= window_time:
            reward = 100.0
        # elif lst_cDELAY > cDELAY and cDELAY <= window_time * 2:
        #     reward = 0.5
        # elif lst_cDELAY >= cDELAY and state < self._minqos:
        #     reward = 1.0
        # elif lst_thpt_var > thpt_var and state < self._minqos:
        #     reward = 1.0
        # elif lst_cDELAY > cDELAY:
        elif cDELAY > window_time:
            reward = -100.0
        else:
            reward = 0.0

        # if (cDELAY > lst_cDELAY) and (cDELAY > window_time):
        #     reward = -100.0




        reward = np.round(reward * 100) / 100
        # reward = np.round(reward * 200)
        # reward = np.clip(reward, a_min=0.0, a_max=1.0)
        # reward = np.clip(reward, a_min=-1.0, a_max=1.0)
        # reward = np.clip(reward, a_min=-100.0, a_max=1.0)
        # print('r_thpt_glo: {}\nr_thpt_var: {}\nr_cDELAY: {}\nr_cTIMEP: {}\nr_RecSparkTotal: {}\nr_RecMQTotal: {}\nr_state: {}\nr_mem_use: {}\nReward: {}'.format(r_thpt_glo, r_thpt_var, r_cDELAY, r_cTIMEP, r_RecSparkTotal, r_RecMQTotal, r_state, r_mem_use, reward))
        self._rewards += reward
        print('** Reward: {}\n** Total Rewards: {}'.format(reward, self._rewards))

        with open(CSV_FILE, mode='a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([thpt_glo, thpt_var, cDELAY, cTIMEP, RecSparkTotal, RecMQTotal, state, mem_use,r_thpt_glo, r_thpt_var, r_cDELAY, r_cTIMEP, r_RecSparkTotal, r_RecMQTotal, r_state, r_mem_use, reward])

        rewards = np.array([r_thpt_glo, r_thpt_var, r_cDELAY, r_cTIMEP, r_RecSparkTotal, r_RecMQTotal, r_state, r_mem_use])
       
        return tf.convert_to_tensor(rewards, dtype=tf.float32)

    # THPT_VAR
    def r_thpt_var_lin_norm(self, thpt_var):
        if(thpt_var > 0 ):
            r_thpt_var = 1
        elif(thpt_var == 0):
            r_thpt_var = 0.5
        else:
            r_thpt_var = 0
        return r_thpt_var
    
    # THPT_GLO
    def r_thpt_glo_lin_norm(self, thpt_glo):
        self._max_thpt = max(self._max_thpt, thpt_glo)
        r_thpt_glo = thpt_glo / self._max_thpt
        r_thpt_glo = np.clip(r_thpt_glo, a_min=0.0, a_max=1.0)
        return r_thpt_glo

    def r_thpt_glo_lin_norm_avg(self, thpt_glo):
        self._max_thpt = (self._max_thpt + thpt_glo)/2
        r_thpt_glo = thpt_glo / self._max_thpt
        r_thpt_glo = np.clip(r_thpt_glo, a_min=0.0, a_max=1.0)
        return r_thpt_glo

    def r_mem_use_lin_norm(self, mem_use):
        r_mem_use = 0.0
        if mem_use > self._maxqos or mem_use < self._minqos:
            r_mem_use = 0.0
        else:
            r_mem_use = (mem_use - self._minqos) / (self._maxqos - self._minqos)
            r_mem_use = np.clip(r_mem_use, a_min=0.0, a_max=1.0)
        return r_mem_use
    
    #  MEM_USE = QoSBase
    def r_mem_use_lin_norm_Inverted(self, mem_use):
        r_mem_use = 0.0
        if mem_use > self._maxqos or mem_use < self._minqos:
            r_mem_use = 0.0
        else:
            r_mem_use = 1 - ((mem_use - self._minqos) / (self._maxqos - self._minqos))
            r_mem_use = np.clip(r_mem_use, a_min=0.0, a_max=1.0)
        return r_mem_use

    def r_mem_use_lin_mid_point(self, mem_use):
        r_mem_use = 0.0
        if mem_use > self._maxqos or mem_use < self._minqos:
            r_mem_use = 0.0
        else:
            midpoint = (self._maxqos + self._minqos) / 2
            r_mem_use = 1 - abs(mem_use - midpoint) / (midpoint - self._minqos)
        return r_mem_use


    # Scheduling Delay
    def r_cDELAY_lin_norm_Inverted_original(self, cDELAY):
        r_cDELAY = 1 - (cDELAY / self._max_cDELAY)
        r_cDELAY = np.clip(r_cDELAY, a_min=0.0, a_max=1.0)
        return r_cDELAY
    
    def r_cDELAY_lin_norm_Inverted(self, cDELAY):
        self._max_cDELAY = max(self._max_cDELAY, cDELAY)
        r_cDELAY = 1 - (cDELAY / self._max_cDELAY)
        r_cDELAY = np.clip(r_cDELAY, a_min=0.0, a_max=1.0)
        return r_cDELAY

    def r_cDELAY_lin_norm_Inverted_avg(self, cDELAY):
        epsilon = 1e-10
        self._max_cDELAY = (self._max_cDELAY + cDELAY) / 2
        r_cDELAY = 1 - (cDELAY / self._max_cDELAY + epsilon)
        r_cDELAY = np.clip(r_cDELAY, a_min=0.0, a_max=1.0)
        return r_cDELAY

    # Processing Delay
    def r_cTIMEP_lin_norm(self, cTIMEP):
        self._max_cTIMEP = max(self._max_cTIMEP, cTIMEP)
        r_cTIMEP = (cTIMEP / self._max_cTIMEP)
        r_cTIMEP = np.clip(r_cTIMEP, a_min=0.0, a_max=1.0)
        return r_cTIMEP
    
    def r_cTIMEP_lin_norm_Inverted(self, cTIMEP):
        self._max_cTIMEP = max(self._max_cTIMEP, cTIMEP)
        r_cTIMEP = 1 - (cTIMEP / self._max_cTIMEP)
        r_cTIMEP = np.clip(r_cTIMEP, a_min=0.0, a_max=1.0)
        return r_cTIMEP
    
    def r_cTIMEP_lin_norm_Inverted_avg(self, cTIMEP):
        epsilon = 1e-10
        self._max_cTIMEP = (self._max_cTIMEP + cTIMEP + epsilon) / 2
        r_cTIMEP = 1 - (cTIMEP / self._max_cTIMEP + epsilon)
        r_cTIMEP = np.clip(r_cTIMEP, a_min=0.0, a_max=1.0)
        return r_cTIMEP

    # STATE
    def r_state_lin_norm(self, state):
        r_state = 0.0
        if state > self._maxqos or state < self._minqos:
            r_state = 0.0
        else:
            r_state = (state - self._minqos) / (self._maxqos - self._minqos)
            r_state = np.clip(r_state, a_min=0.0, a_max=1.0)
        return r_state

    def r_state_lin_norm_Inverted(self, state):
        r_state = 0.0
        if state > self._maxqos or state < self._minqos:
            r_state = 0.0
        else:
            r_state = 1 - (state - self._minqos) / (self._maxqos - self._minqos)
            r_state = np.clip(r_state, a_min=0.0, a_max=1.0)
        return r_state
    
    def r_state_lin_norm_Inverted_step(self, state):
        r_state = 0.0
        if state > self._maxqos or state < self._minqos:
            r_state = 0.0
        else:
            r_state = 0.3 * (1 - (state - self._minqos) / (self._maxqos - self._minqos))
            r_state = np.clip(r_state, a_min=0.0, a_max=0.3)
        return r_state

    def r_state_lin_mid_point(self, state):
        r_state = 0.0
        if state > self._maxqos or state < self._minqos:
            r_state = 0.0
        else:
            midpoint = (self._maxqos + self._minqos) / 2
            r_state = 1 - abs(state - midpoint) / (midpoint - self._minqos)
        return r_state
    
    def r_state_lin_mid_point_step(self, state):
        r_state = 0.0
        if state > self._maxqos or state < self._minqos:
            r_state = 0.0
        else:
            midpoint = (self._maxqos + self._minqos) / 2
            r_state = 1 - abs(state - midpoint) / (midpoint - self._minqos)
            # Round to the nearest 0.2
            r_state = round(r_state * 5) / 5
            r_state = np.clip(r_state, a_min=0.0, a_max=1.0)
        return r_state

    def r_RecSparkTotal_reward(self, RecSparkTotal, lst_RecSparkTotal):
        return 1 if RecSparkTotal >= lst_RecSparkTotal else 0

    def r_RecMQTotal_reward(self, RecMQTotal, lst_RecMQTotal):
        return 1 if RecMQTotal >= lst_RecMQTotal else 0



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

        self.ppo_agent.addToBuffer(last_time_step, last_action, current_time_step)
        self.ppo_agent.train()
        
        # traj = trajectory.from_transition(last_time_step, last_action, current_time_step)
        # traj_batched = tf.nest.map_structure(lambda t: tf.stack([t] * 1), traj)
        # self.ppo_agent.addToBuffer(traj_batched)

        
        # if (self.ppo_agent.replay_buffer.num_frames().numpy() > GLOBAL_STEPS * GLOBAL_BATCH):
        # if not (self.ppo_agent.replay_buffer.num_frames().numpy() % GLOBAL_STEPS):
            # if self.ppo_agent.replay_buffer.num_frames().numpy() % (self._batch_size * 4):
            # if not (self.ppo_agent.replay_buffer.num_frames().numpy() % (self._batch_size/2)):
            # if not (self.ppo_agent.replay_buffer.num_frames().numpy() % self._batch_size):
            # if not (self.ppo_agent.replay_buffer.num_frames().numpy() % GLOBAL_STEPS):
            # if not (self.ppo_agent.replay_buffer.num_frames().numpy() > GLOBAL_STEPS):
            #     if not (self.ppo_agent.replay_buffer.num_frames().numpy() % GLOBAL_STEPS/2):
            # self.ppo_agent.train()
            
        self._last_action = self.ppo_agent.getAction(current_time_step)
       
        # # Return action -1 because the actions are mapped to 0,1,2 need to -> 1, 0, 1
        #     action = self._last_action.action.numpy() - 1
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
    return action

cdef public void finish(object agent, float* last_state):
    state = []
    for i in range(8):
        state.append(last_state[i])
    
    agent.finish(state)
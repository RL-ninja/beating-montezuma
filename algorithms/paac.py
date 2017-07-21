import time
import numpy as np
import logging
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float

from algorithms.actor_learner import *
from environments.emulator_runner import EmulatorRunner
from environments.runners import Runners

        
class PAACLearner(ActorLearner):
    def __init__(self, network_creator, environment_creator, args): 
        super(PAACLearner, self).__init__(network_creator, environment_creator, args)
        self.workers = args.emulator_workers
        self._imd_vars = {}

    @staticmethod
    def choose_next_actions(network, num_actions, states, session):
        # network_output_pi: softmax output that gives a prob dist.
        network_output_v, network_output_pi = session.run(
            [network.output_layer_v,
             network.output_layer_pi],
            feed_dict={network.input_ph: states})
        action_indices = PAACLearner.__sample_policy_action(network_output_pi)

        new_actions = np.eye(num_actions)[action_indices]

        return new_actions, network_output_v, network_output_pi

    def _choose_next_actions(self, states):
        return PAACLearner.choose_next_actions(self.network, self.num_actions, states, self.session)

    @staticmethod
    def __sample_policy_action(probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        # TODO: add epsilon greedy logic here
        probs = probs - np.finfo(np.float32).epsneg
                
        # Bolztmann exploration: sampling by the prob distribution.
        # note we don't argmax (=greedy) here to bring in diversity.
        action_indexes = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probs]
        return action_indexes

    def _get_shared(self, array, dtype=c_float):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :param dtype: the RawArray dtype to use
        :return: the RawArray backed numpy array
        """
        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    
    def _init_shared_vars(self):
        # state, reward, episode_over(terminated), action, bonus
        variables = [(np.asarray([emulator.get_initial_state() for emulator in self.emulators], dtype=np.float32)),
                     (np.zeros(self.emulator_counts, dtype=np.float32)),
                     (np.asarray([False] * self.emulator_counts, dtype=np.float32)),
                     (np.zeros((self.emulator_counts, self.num_actions), dtype=np.float32)),
                     (np.zeros(self.emulator_counts, dtype=np.float32))]
        return variables


    def _init_environments(self, variables):
        self.runners = Runners(EmulatorRunner, self.emulators, self.workers, variables)
        self.runners.start()


    def _init_train(self):
        # initialize network
        self.global_step = self.init_network()
        self.global_step_start = self.global_step
        logging.debug("Starting training at Step {}".format(self.global_step))
        
        variables = self._init_shared_vars()
        # initialize environments and workers
        self._init_environments(variables)        
        self.total_rewards = []
        self.summaries_op = tf.summary.merge_all()
        
        # initialize intermediate vars    
        self._init_intermediate_vars()


    def _should_training_continue(self):
        return self.global_step < self.max_global_steps
    

    def _init_intermediate_vars(self):
        # load intermediate/non-global variables
        shared_states, shared_rewards, shared_episode_over, shared_actions, shared_bonuses = self.runners.get_shared_variables()

        emulator_steps = [0] * self.emulator_counts
        total_episode_rewards = self.emulator_counts * [0]

        actions_sum = np.zeros((self.emulator_counts, self.num_actions))
        y_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        adv_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        rewards = np.zeros((self.max_local_steps, self.emulator_counts))
        states = np.zeros([self.max_local_steps] + list(shared_states.shape), dtype=np.float32)
        actions = np.zeros((self.max_local_steps, self.emulator_counts, self.num_actions))
        values = np.zeros((self.max_local_steps, self.emulator_counts))
        episodes_over_masks = np.zeros((self.max_local_steps, self.emulator_counts))
        start_time = time.time()

        self._imd_vars['shared_states'] = shared_states
        self._imd_vars['shared_rewards'] = shared_rewards
        self._imd_vars['shared_episode_over'] = shared_episode_over
        self._imd_vars['shared_actions'] = shared_actions
        self._imd_vars['shared_bonuses'] = shared_bonuses
        self._imd_vars['emulator_steps'] = emulator_steps
        self._imd_vars['total_episode_rewards'] = total_episode_rewards
        self._imd_vars['actions_sum'] = actions_sum
        self._imd_vars['y_batch'] = y_batch
        self._imd_vars['adv_batch'] = adv_batch
        self._imd_vars['rewards'] = rewards
        self._imd_vars['states'] = states
        self._imd_vars['actions'] = actions
        self._imd_vars['values'] = values
        self._imd_vars['episodes_over_masks'] = episodes_over_masks
        self._imd_vars['start_time'] = start_time
        self._imd_vars['counter'] = 0

    def _run_actors(self):
        shared_states = self._imd_vars['shared_states']
        shared_rewards = self._imd_vars['shared_rewards']
        shared_episode_over = self._imd_vars['shared_episode_over']
        shared_actions = self._imd_vars['shared_actions']
        emulator_steps = self._imd_vars['emulator_steps']
        total_episode_rewards = self._imd_vars['total_episode_rewards']
        actions_sum = self._imd_vars['actions_sum']
        rewards = self._imd_vars['rewards']
        states = self._imd_vars['states']
        actions = self._imd_vars['actions']
        values = self._imd_vars['values']
        episodes_over_masks = self._imd_vars['episodes_over_masks']

        for t in range(self.max_local_steps):
            next_actions, readouts_v_t, _ = self._choose_next_actions(shared_states)
            
            actions_sum += next_actions
            
            for env_i in range(next_actions.shape[0]):
                # for simplicty, we keep one global shared_vars object
                # in sync, not multiple vars for each worker
                shared_actions[env_i] = next_actions[env_i]

            actions[t] = next_actions
            values[t] = readouts_v_t
            states[t] = shared_states

            # Start updating all environments with next_actions
            self.runners.update_environments()
            self.runners.wait_updated()
            # Done updating all environments, have new states, rewards and is_over

            episodes_over_masks[t] = 1.0 - shared_episode_over.astype(np.float32)

            for env_i, (raw_reward, episode_over) in enumerate(zip(shared_rewards, shared_episode_over)):                
                # reward for statistics
                total_episode_rewards[env_i] += raw_reward
                # reward for training
                r = self.clip_reward(raw_reward)
                rewards[t, env_i] = r
                                
                emulator_steps[env_i] += 1                                    
                self.global_step += 1
                
                if episode_over:
                    self.total_rewards.append(total_episode_rewards[env_i])
                    self._update_tf_summary(env_i)
                    total_episode_rewards[env_i] = 0
                    emulator_steps[env_i] = 0
                    actions_sum[env_i] = np.zeros(self.num_actions)


    def _run_learners(self):    
        shared_states = self._imd_vars['shared_states']
        rewards = self._imd_vars['rewards']
        states = self._imd_vars['states']
        actions = self._imd_vars['actions']
        values = self._imd_vars['values']
        episodes_over_masks = self._imd_vars['episodes_over_masks']
        y_batch = self._imd_vars['y_batch']
        adv_batch = self._imd_vars['adv_batch']

        nest_state_value = self.session.run(
            self.network.output_layer_v,
            feed_dict={self.network.input_ph: shared_states})

        estimated_return = np.copy(nest_state_value)

        for t in reversed(range(self.max_local_steps)):
            # if terminated, no estimated_return
            estimated_return = rewards[t] + self.gamma * estimated_return * episodes_over_masks[t]
            y_batch[t] = np.copy(estimated_return)
            adv_batch[t] = estimated_return - values[t]

        flat_states = states.reshape([self.max_local_steps * self.emulator_counts] + list(shared_states.shape)[1:])
        flat_y_batch = y_batch.reshape(-1)
        flat_adv_batch = adv_batch.reshape(-1)
        flat_actions = actions.reshape(self.max_local_steps * self.emulator_counts, self.num_actions)

        lr = self.get_lr()
        feed_dict = {self.network.input_ph: flat_states,
                     self.network.critic_target_ph: flat_y_batch,
                     self.network.selected_action_ph: flat_actions,
                     self.network.adv_actor_ph: flat_adv_batch,
                     self.learning_rate: lr}

        _, summaries = self.session.run(
            [self.train_step, self.summaries_op],
            feed_dict=feed_dict)

        self.summary_writer.add_summary(summaries, self.global_step)
        self.summary_writer.flush()


    def _update_tf_summary(self, env_i):
        emulator_steps = self._imd_vars['emulator_steps']
        total_episode_rewards = self._imd_vars['total_episode_rewards']
        episode_summary = tf.Summary(value=[
            tf.Summary.Value(tag='rl/reward', simple_value=total_episode_rewards[env_i]),
            tf.Summary.Value(tag='rl/episode_length', simple_value=emulator_steps[env_i]),
        ])
        self.summary_writer.add_summary(episode_summary, self.global_step)
        self.summary_writer.flush()


    def _log_training(self, loop_start_time):
        counter = self._imd_vars['counter']
        start_time = self._imd_vars['start_time']
        if counter % (2048 / self.emulator_counts) == 0:
            curr_time = time.time()
            global_steps = self.global_step
            last_ten = 0.0 if len(self.total_rewards) < 1 else np.mean(self.total_rewards[-10:])
            msg = "Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
            logging.info(msg.format(global_steps,
                                 self.max_local_steps * self.emulator_counts / (curr_time - loop_start_time),
                                 (global_steps - self.global_step_start) / (curr_time - start_time),
                                 last_ten))

    def train(self):
        """
        Main actor learner loop for parallel advantage actor critic learning.
        """
        self._init_train()

        while self._should_training_continue():
            loop_start_time = time.time()
            self._run_actors()
            self._run_learners()
            
            self._imd_vars['counter'] += 1
            self._log_training(loop_start_time)
            self.save_vars()

        self.cleanup()

    def cleanup(self):
        super(PAACLearner, self).cleanup()
        self.runners.stop()
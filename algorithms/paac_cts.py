'''
CTS reference implmentations:
https://github.com/brendanator/atari-rl/blob/master/agents/exploration_bonus.py
https://github.com/steveKapturowski/tensorflow-rl/blob/master/algorithms/intrinsic_motivation_actor_learner.py
https://github.com/steveKapturowski/tensorflow-rl/blob/master/utils/fast_cts.pyx
PixelCNN
'''
# from actor_learner import ONE_LIFE_GAMES

import numpy as np
import time
import logging
    
from algorithms.paac import PAACLearner
from environments.emulator_runner import EmulatorRunner
from environments.runners import Runners
from utilities.cts.cts_mixin import CTSDensityModelMixin


class CTSEmulatorRunner(EmulatorRunner, CTSDensityModelMixin):
    """docstring for CTSEmulatorRunner"""
    def __init__(self, *args):
        super(CTSEmulatorRunner, self).__init__(*args)
        self._init_density_model()    

    def _run(self):
        count = 0
        while True:
        
            instruction = self.queue.get()
            if instruction is None:
                break
            
            shared_states = self.variables[0]
            shared_rewards = self.variables[1]
            shared_episode_over = self.variables[2]
            shared_actions = self.variables[3]
            shared_bonuses = self.variables[4]
            
            for i, (emulator, action) in enumerate(zip(self.emulators, shared_actions)):            
                emulator = self.emulators[i]
                new_s, reward, episode_over = emulator.next(action)
                
                bonus = self.get_exploration_bonus(new_s)
                shared_bonuses[i] = bonus

                if episode_over:
                    shared_states[i] = emulator.get_initial_state()
                else:
                    shared_states[i] = new_s            
                shared_rewards[i] = reward                
                shared_episode_over[i] = episode_over
            
            count += 1
            
            # barrier is a queue shared by all workers
            # when a worker is done executing actions for envs it manages
            # it puts True to barrier which later should be 
            self.barrier.put(True)


class PseudoCountPAACLearner(PAACLearner):
    def __init__(self, network_creator, environment_creator, args):
        super(PseudoCountPAACLearner, self).__init__(network_creator, environment_creator, args)


    def _init_environments(self, variables):
        self.runners = Runners(CTSEmulatorRunner, self.emulators, self.workers, variables)
        self.runners.start()

    def _run_actors(self):
        shared_states = self._imd_vars['shared_states']
        shared_rewards = self._imd_vars['shared_rewards']
        shared_episode_over = self._imd_vars['shared_episode_over']
        shared_actions = self._imd_vars['shared_actions']
        shared_bonuses = self._imd_vars['shared_bonuses']

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
            for env_i, (raw_reward, bonus, episode_over) in enumerate(zip(shared_rewards, shared_bonuses, shared_episode_over)):                
                total_episode_rewards[env_i] += raw_reward
                rewards[t, env_i] = self.clip_reward(raw_reward + bonus)

                emulator_steps[env_i] += 1                                    
                self.global_step += 1
                
                # save and sync density model every 0.2M steps
                SAVE_INTERVAL = 1000000

                if self.global_step % SAVE_INTERVAL == 0:
                    runners = self.runners.runners
                    master_model = runners[0]._density_model # arbitrary
                    CTSDensityModelMixin.save_density_model(master_model)
                
                if episode_over:
                    self.total_rewards.append(total_episode_rewards[env_i])
                    self._update_tf_summary(env_i)
                    total_episode_rewards[env_i] = 0
                    emulator_steps[env_i] = 0
                    actions_sum[env_i] = np.zeros(self.num_actions)

  
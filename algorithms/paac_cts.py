'''
CTS reference implmentations:
https://github.com/brendanator/atari-rl/blob/master/agents/exploration_bonus.py
https://github.com/steveKapturowski/tensorflow-rl/blob/master/algorithms/intrinsic_motivation_actor_learner.py
https://github.com/steveKapturowski/tensorflow-rl/blob/master/utils/fast_cts.pyx
PixelCNN
'''
# from actor_learner import ONE_LIFE_GAMES

import numpy as np
# import cPickle

from algorithms.paac import PAACLearner
from utilities.fast_cts import CTSDensityModel


class PAACCTSLearner(PAACLearner):
    """docstring for PAACCTSLearner"""
    def __init__(self, network_creator, environment_creator, args):
        super(PAACCTSLearner, self).__init__(network_creator, environment_creator, args)

        # sec. 3.3 in https://arxiv.org/pdf/1703.01310v2.pdf says
        # a good combination is: lr=0.001, c=0.1
        # and we don't use beta at all
        # note the reported setup is for pixelcnn
        # a large c slows decay rate
        model_args = {
            'height': 42,
            'width': 42,
            'num_bins': 8,
            'c': 0.1,
            'n': self.global_step
        }

        self._density_model = CTSDensityModel(**model_args)


    def _compute_bonus(self, state):
        # TODO: decouple _density_model.update into update and compute bonus 
        state = state[:,:, -1].astype(np.float32)
        state /= 256.0 # 8 bit grayscale/rgb
        return self._density_model.update(state)


    def _save_density_model(self):
        logger.info('T{} Writing Pickled Density Model to File...'.format(''))
        raw_data = cPickle.dumps(self.density_model.get_state(), protocol=2)
        with self.barrier.counter.lock, open('/tmp/density_model.pkl', 'wb') as f:
            f.write(raw_data)

        self.should_density_model_sync = True


    def _sync_density_model(self):
        logger.info('T{} Synchronizing Density Model...'.format(self.actor_id))
        with self.barrier.counter.lock, open('/tmp/density_model.pkl', 'rb') as f:
            raw_data = f.read()

        self.density_model.set_state(cPickle.loads(raw_data))


    def _process_reward(self, t, env_i, raw_reward):
        total_episode_rewards = self._imd_vars['total_episode_rewards']
        states = self._imd_vars['states']
        rewards = self._imd_vars['rewards']

        total_episode_rewards[env_i] += raw_reward
        r = self.clip_reward(raw_reward)
        # bonus = self._compute_bonus(states[t][env_i])
        bonus = 0
        rewards[t, env_i] = r + bonus

        # if self.global_step % self.density_model_update_steps == 0:
        #     self.save_density_model()
        # if self.should_density_model_sync:
        #     self.sync_density_model()
        #     self.should_density_model_sync = False
              


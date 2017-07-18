'''
CTS reference implmentations:
https://github.com/brendanator/atari-rl/blob/master/agents/exploration_bonus.py
https://github.com/steveKapturowski/tensorflow-rl/blob/master/algorithms/intrinsic_motivation_actor_learner.py
https://github.com/steveKapturowski/tensorflow-rl/blob/master/utils/fast_cts.pyx
PixelCNN
'''
# from actor_learner import ONE_LIFE_GAMES

from algorithms.paac import PAACLearner
from utilities.fast_cts import CTSDensityModel
import numpy as np

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
            'global_step': self.global_step
        }

        self._density_model = CTSDensityModel(**model_args)

    def _compute_bonus(self, state):
        state = state[:,:, -1].astype(np.float32)
        state /= 256.0 # 8 bit grayscale/rgb
        return self._density_model.update(state)


    def rescale_reward(self, reward, state):
        """ Clip raw reward """
        if reward > 1.0:
            reward = 1.0
        elif reward < -1.0:
            reward = -1.0        
        bonus = self._compute_bonus(state)
        reward += bonus
        print(bonus)
        return reward
'''
CTS reference implmentations:
https://github.com/brendanator/atari-rl/blob/master/agents/exploration_bonus.py
https://github.com/steveKapturowski/tensorflow-rl/blob/master/algorithms/intrinsic_motivation_actor_learner.py
https://github.com/steveKapturowski/tensorflow-rl/blob/master/utils/fast_cts.pyx
PixelCNN
'''
# from actor_learner import ONE_LIFE_GAMES

from algorithms.paac import PAACLearner
# original CTS version
# from utilities.cts_density_model import CTSDensityModel
# faster CTS version
from utilities.fast_cts import CTSDensityModel
# original CTS version#2 (logic is the same)
# from utilities.cts_bonus import ExplorationBonus
import numpy as np

class PAACCTSLearner(PAACLearner):
    """docstring for PAACCTSLearner"""
    def __init__(self, network_creator, environment_creator, args):
        super(PAACCTSLearner, self).__init__(network_creator, environment_creator, args)


        model_args = {
            'height': 42,
            'width': 42,
            'num_bins': 8,
            'beta': 0.05
        }

        self._density_model = CTSDensityModel(**model_args)
        # self.bonuses = []
        # self._density_model = ExplorationBonus(**model_args)

    def _compute_bonus(self, state):
        latest_state = state[:,:, -1]
        # import pdb;pdb.set_trace()
        return self._density_model.update(latest_state)
        # return self._density_model.bonus(latest_state)


    def rescale_reward(self, reward, state):
        """ Clip raw reward """
        if reward > 1.0:
            reward = 1.0
        elif reward < -1.0:
            reward = -1.0        
        bonus = self._compute_bonus(state)
        reward += bonus
        return reward
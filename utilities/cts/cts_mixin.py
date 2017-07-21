import numpy as np
import pickle

from os import path

from utilities.cts.fast_cts import CTSDensityModel

DENSITY_MODEL_FILEPATH = '/tmp/density_model.pkl'


class CTSDensityModelMixin(object):
    """docstring for CTSDensityModelMixin"""
    
    def __init__(self, *arg):
        super(CTSDensityModelMixin, self).__init__()        
    
    def _init_density_model(self, height=42, width=42, num_bins=8, beta=0.05):
        args = {
            'height': height,
            'width': width,
            'num_bins': num_bins,
            'beta': beta
        }
        self._density_model = CTSDensityModel(**args)
        if path.isfile(DENSITY_MODEL_FILEPATH):
            # TODO: should check for game name ... 
            self.sync_density_model()
    
    def get_exploration_bonus(self, obs):
        obs = obs[..., -1]
        # state will be quantized to 3 bit grayscale later (b/c num_bins)
        return self._density_model.update(obs)
        # return 0.01

    def sync_density_model(self):
        print('Synchronizing Density Model...'.format(''))
        with open(DENSITY_MODEL_FILEPATH, 'rb') as f:
            raw_data = f.read()

        self._density_model.set_state(pickle.loads(raw_data))

    
    @staticmethod    
    def update_density_model(model, states):
        # preprocessing batch
        obs_batch = np.asarray(states[..., -1])
        return model.update(obs_batch)


    @staticmethod
    def save_density_model(model):
        print('Writing Pickled Density Model to File...'.format(''))
        raw_data = pickle.dumps(model.get_state(), protocol=2)
        with open(DENSITY_MODEL_FILEPATH, 'wb') as f:
            f.write(raw_data)

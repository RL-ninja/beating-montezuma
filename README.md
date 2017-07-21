# Beating Montezuma's Revenge
We aim to achieve the state-of-the art results on [Montezuma's Revenge](https://github.com/RL-ninja/beating-montezuma/wiki/Montezuma's-Revenge

## TODOs & Ideas
See [Todos & Ideas](https://github.com/RL-ninja/beating-montezuma/wiki/TODOs-and-Ideas)

## Evaluation Metrics
- [ ] game score:
	* [ ] average score > [3500](https://gym.openai.com/envs/MontezumaRevenge-v0) (of last 10 episodes)
	* [ ] max score > [6600](https://www.youtube.com/watch?v=EzQwCmGtEHs&feature=youtu.be)  
- [ ] number of rooms/level explored:
	* [ ] 20+ rooms
	* [ ] beat level 1

## Resources (papers & implementations)
See [Resources](https://github.com/RL-ninja/beating-montezuma/wiki/Resources)

## Getting started
### Dependencies
If you use Anaconda, you can try ```conda env create -f environment.yml```.

Requirements
* Python 3.4+
* TensorFlow 1.0+ (choose a GPU version, if you have GPU)
* [Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
* cython (pip3 package)
* scikit-image (pip3 package)
* python3-tk
* opencv (opencv-python)

### Training the agent
To train an agent to play, for example, pong run
* ```python3 train.py -g <game-name> -df logs/<game-name>/ -algo paac_cts```

### Visualizing training
1. Open a new terminal
2. Attach to the running docker container with ```docker exec -it CONTAINER_NAME bash```
3. Run ```tensorboard --logdir=<absolute-path>/paac/logs/tf```.
4. In your browser navigate to localhost:6006/

If running locally, skip step 2.

### Testing the agent
To test the performance of a trained agent run ```python3 test.py -f logs/ -tc 5```
Output:
```
Performed 5 tests for seaquest.
Mean: 1704.00
Min: 1680.00
Max: 1720.00
Std: 14.97
```

### Generating gifs
```
python3 test.py -f logs/<game-name>/ -gn breakout
```
This may take a few minutes.

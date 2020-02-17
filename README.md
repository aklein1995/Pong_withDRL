# Playing PONG

Solving the PONG game using REINFORCE and PPO algorithms

## Getting Started
The provided code is developed with Python 3 and PyTorch. Moreover, other packages such as numpy, matplotlib, gym and multiprocess are required.

### Dependencies
To play PONG we are going to use the environment provided by OpenAI. Hence, we would have to download as:
- __OpenAI PONG__:
```bash
pip install gym[atari]
```
Moreover, to check the progress it is required to download the next pip packages (JSAnimation it is not strictly required)
- __Monitorization__:
```bash
pip install JSAnimation
pip install progressbar
```

## Instructions
The code could be executed either from any of the provided notebooks or from the PlayingPong.py script.  
In the latter case, at the init of the script it would be necessary to specify the hyperparameter tunning and the algorith that we wanna use for training.

## Results
- __REINFORCE__:

- __PPO__:

## Acknowledgements
The code provided in the Udacity DRLND has been used as baseline in order to adapt and solve this problem.

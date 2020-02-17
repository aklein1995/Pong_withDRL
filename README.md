[image1]: https://user-images.githubusercontent.com/25618603/74645299-ee592880-5177-11ea-80a0-35cb51b2646d.png "Trained REINFORCE"
[image2]: https://user-images.githubusercontent.com/25618603/74649048-1c8e3680-517f-11ea-8687-a9d7d10eca2c.png "Trained PPO"

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
The code could be executed either from any of the provided notebooks or from the `PlayingPong.py` script.  
In the latter case, at the init of the script it would be necessary to specify the hyperparameters values and the algorith that we wanna use for training.

## Results
Both agents seem to learn with only 500 episodes, although PPO converges faster to better results.

- __REINFORCE__:  
![Trained REINFORCE][image1]
- __PPO__:  
![Trained PPO][image2]

## Acknowledgements
The code provided in the Udacity DRLND has been used as baseline in order to adapt and solve this problem.

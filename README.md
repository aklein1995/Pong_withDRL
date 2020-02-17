[image1]: https://user-images.githubusercontent.com/25618603/74645299-ee592880-5177-11ea-80a0-35cb51b2646d.png "Trained REINFORCE"
[image2]: https://user-images.githubusercontent.com/25618603/74649048-1c8e3680-517f-11ea-8687-a9d7d10eca2c.png "Trained PPO"

# Playing PONG

Solving the PONG game using REINFORCE and PPO algorithms

We would train our agent from raw pixels. In order to do that, we would use a neural network architecture composed of Convolutional and Linear layers, all of them sequantilly stacked. Moreover, to make the learning process easiear, the input frames are preprocessed reducing its dimensionality and the possible colors (changing it to grey scale).
![image](https://user-images.githubusercontent.com/25618603/74650293-a6d79a00-5181-11ea-8152-fbaa14d8bef6.png)
  
The NN would only have one neuron, corresponding to take `RIGHT` action, whose probability would be calculated with a softmax function. The other possible actions, `LEFT` would be computed as 1-p(RIGHT).

## Getting Started
The provided code is developed with Python 3 and PyTorch. Moreover, other packages such as numpy, matplotlib, gym and multiprocess are required.

The easiest way to get started would be to import the conda environment that I have exported and uploaded to this repository.
- __Linux, Ubuntu 18.04 LTS__:
```bash
conda env create -f drlnd_pong.yml
```  
If you use another OS or you have any issue during the process, follow the next steps.

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

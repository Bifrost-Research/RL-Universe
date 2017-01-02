# What is in this repo?
Implementation of a Reinforcement Learning algorithms in Python3 using he OpenAI framework and TensorFlow as neural network libraries. Results are shown with TensorBoard.

#Gym-A3C
Implementation of the A3C algorithm for OpenAI Gym's Atari games. The implementation uses processes instead of threads to achieve real concurrency. 

##How to run the A3C ?
To lauch the A3C with the default parameters, just use the following command. It is possible to see the available hyper parameters with the command -h.
```
python3.5 main.py Pong-v0
```
To see different plots like the rewards, the losses... it is necessary to launch TensorBoard with the command
```
tensorboard --logdir=/tf_logs/
```

##Results

Pong-v0, 4 actors, 5 local steps and updating the network with an Adam Optimizer (learning rate: 1e-4).

#Useful papers
[A3C](https://arxiv.org/abs/1602.01783)

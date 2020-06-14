[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition

### Introduction

This folder contains my solution for Udacity's Deep Reinfrocement Learning Nanodegree third project. In this task an agents should learn how to control a racket to bounce a ball over a net (tennis). This enviroment was developed in Unity with the [Machine Learning Agents Package](https://github.com/Unity-Technologies/ml-agents.git)  


![Trained Agent][image1]

## Environment

The environment consist on `2` agents with a racket each one. For this project the agents must lear how to move the racket and bounce the ball over the net so the ball doesn't reach the floor. Therefore the goal of each agent is to know how maximize the number of times that each racket bounce the ball over the net and keep the ball in play.

**Imporant Note:** The goal of the agents is to maximaze the playing time, not to win the match.

### State Space

The environment has a continous `8` dimensional state space corresponding to the position and velocity of the ball and its own racket. 

### Action space

Each agent has a continous `2` dimensional action space, corresponding to the movement of the racket and jumping. Due to the definition of the environment, each action in the action vector, should be a number between `-1` and `1`.

### Rewards

- `+0.1` If an agent hits the ball over the net.
- `-0.1` If an agent lets the ball hits the ground or hits the ball out of bounds.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

## Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  

## Credit

The following project was solved using most of the code that was provided by [Udacity's Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning) for DDPG Learning.


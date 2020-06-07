[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Udacity's Deep Reinforcement Learning Nanodegree - Project 2: Continuous Control

## Introduction

This folder contains my solution for Udacity's Deep Reinfrocement Learning Nanodegree second project. In this task an agent should learn how to articulate its arm to maintain its hand in a target location as many time steps possible. This enviroment was developed in Unity with the [Machine Learning Agents Package](https://github.com/Unity-Technologies/ml-agents.git)  

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

## Environment

The environment consist on `20` agents that have the form of an arm. For this project the agents must lear how to move their joints to mantain their hand in a desired location. Therefore the goal of each agent is to maximize the time steps that the hand is in a target location.

### State Space

The environment has a continous `33` dimensional state space corresponding to the position, rotation, velocity, and angular velocities of the arm of each agent.

### Action space

Each agent has a continous `4` dimensional action space, corresponding to the torque applicable to two joints. Due to the definition of the environment, each action in the action vector, should be a number between `-1` and `1`.

### Rewards

A reward of `+0.1` is provided for each step that the agent's hand is in the goal location.

## Getting Started

**Please make sure that you followed the instalation instructions in [Udacity-DeepRL-Nanodegree](https://github.com/wmoralesap/Udacity-DeepRL-Nanodegree)**

### Download Unity Environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the current GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!  

## Credit

The following project was solved using most of the code that was provided by [Udacity's Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning) for DDPG Learning.

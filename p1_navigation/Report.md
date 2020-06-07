# Udacity's Deep Reinforcement Learnig - Project 1: Navigation Report

This is the report for the first project in the Udacity's Deep Reinforcement Learning Nanodegree Program. This reports explains the details of the algorithm usen to train an agent to solve play in a given scenario.



## Navigation

The given task was to program an smart agent to navigate around an environment to maximaze the cumulative rewards. This section discusses the details of the environment where the agent is trained.

### Environment

The environment consisted in a square world where there are blue and yellow bananas randomly scattered. The task of the smart agent is to navigate through this environment to maximaze the amount of **yellow** bananas that it collects, while avoiding the blue bananas. The enviroment was developed in Unity using the Machine Learning Agent plugin. This plugin allows developers to create enviroments where they can use to train agents using Reinforcement Learning using it's python API. For this environment the rewards given to the agent are defined as follows.

- `+1` for each **yellow** banana collected by the agent
- `-1` for each **blue** banana collected by the agent

### Action Space

This environment contents a one dimensional discrete action space with **4** possible actions defined as follows: 

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### State Space

The implemented environment contents a 37 continous dimensional space that represented the agent's velocity and raycast based perception of the objects that were in front of the agent. 

## Method

This section discuss the implemented algorithm to train the smart agent.

### DeepQ Learning with Expirience Replay

To train the agent, we implemente the DeepQ Learning algorithm developed by Mnih' *et al* in Google's Mind [paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).

In this algorithm we train a predefined neuronal network to predict the value functions from a give state called Local Network. As in bellman equation, we need to compute the value functions of the next state that will serve as target to learn the optimal value function of a given state. To this end the DQN algorithm estimates the next state value fuction with an identical neuronal network to the local one, called the Target Network. Therefore the update of the local neuronal networks weights is defined as:

<p align="center">
    <img src="https://user-images.githubusercontent.com/27258035/83926222-962b4e00-a789-11ea-8fb2-25393b948601.PNG" width="400"/>
</p>

This Target Network generate the next state value fuctions using the last weights of the local one in the training process for a period of time. After this period, the target weights are updated with the Local ones and we continue training the Local network with the new weights of the target network. This process is repeated until the training process ends. 

To avoid a correlation in the sequence used to train the agent, we implemented experience replay where we store the **`(State, Action, Reward, Next_State)`** tuples in a replay buffer and sample from it a random sequence. Therefore, the DeepQ Learning with Experience Replay algorithm follows the following pseudo-code:

<p align="center">
    <img src="https://user-images.githubusercontent.com/27258035/83926095-2b7a1280-a789-11ea-9860-51f12fecd08e.png" width="600"/>
</p>

In this implementation, we train every `4` steps with a batch size of `64` expiriences. Finally we continued training the agent until the average reward of `100` continues episode was greater than `16` or until we reach `2000` episodes.

### Neuronal Network

To estimate the value fuction from a given state we use a Neuronal Network composed with `2` fully connected hidden layers of `64` units each one, with RELU activation functions and an output layer with softmax activation function.

### Hyperparameters

The agent is trained with the following hyperparameters:

- **`Batch size`**  =  64
- **`Gamma`**  =  0.99
- **`TAU`**  =  1e-3
- **`Learning Rate`**  =  5e-4
- **`Steps per update`**  =  4
- **`Start Epsilon`**  = 1.0
- **`End Epsilon`**  = 0.01
- **`Epsilon Decay`**  = 0.995


## Result

The following figure shows the reward obtained in each episode while training the agent.

<p align="center">
    <img src="https://user-images.githubusercontent.com/27258035/83928119-eb1d9300-a78e-11ea-8395-b7b1fcaca5af.PNG" width="600"/>
</p>

As it can be seen, the agent reached the end criteria after **`897`**  episodes, with a 100 episode average reward of **`16.01`**.

## Improvements

Although we trained an agent to play in the current enviroment with a good average reward per episode, there is still room for improvements. Concretely, we could extract the image of the environment and Dueling Q Networks to get a better performance of the agent. Also we could use Importance Sampling instead of random sampling to learn from action-state pairs that are less probably to happen in the current environment.

# Udacity's Deep Reinforcement Learnig - Project 2: Continous Control  Report

This is the report for the second project in the Udacity's Deep Reinforcement Learning Nanodegree Program. This reports explains the details of the algorithm usen to train an agent to solve play in a given scenario.

## Continous Contronl

The given task was to program `20` smart agents that has to articulate an arm like object to maximaze the time steps that then tip of the object stays in a desired location. This section discusses the details of the environment where the agents is trained.

### Environment

The environment consisted in a open world with `20` arm like objects scattered around the area. The task of the smart agents is to move the joints of the arms (each agent per arm) to mantain the hand in a desired location represented as a ball. The enviroment was developed in Unity using the Machine Learning Agent plugin. This plugin allows developers to create enviroments where they can use to train agents using Reinforcement Learning using it's python API. For this environment each agent receive reward of `+0.1` for each time step that the hand is in the desired location. 

### Action Space

Each agent of the environment contents a four dimensional continous action space corresponding to the torque applicable to the two joints of each arm. In this case, the range of each action space is between `-1` and `1`. As this environment consist of `20` agents. The action space is a matrix of (`20`,`4`)

### State Space

Each agent of the environment has a continous `33` dimensional state space corresponding to the position, rotation, velocity, and angular velocities of the arm of each agent. As this environment consist of `20` agents. The state space is a matrix of (`20`,`33`)

## Method

This section discuss the implemented algorithm to train the smart agent.

### DDPG Learning with Expirience Replay

To train the agent, we implemente the Deep Deterministic Policy Gradient (DDPG) learning technique devolped by Lillicrap *et al* [paper](https://arxiv.org/abs/1509.02971)

This algorithm is a actor-critic method where we train a `2` predefined neuronal networks (actor and critic) to act according to the current states and solved a task. In these methods the critic is in charge to estimate the value function of the given state that is used by the actor to learn and predict the best action. As in DQN, we will have two identical neuronal networks to the actor and the critic that will calculate the targets that agent will learn.

This Actor-Critic Target Networks will generate the target using the last weights of the Local Networks for a period of time. After this period has finished, the Targets weights are updated with the weights of the local networks in that time step, and then the learning process of the local networks is ranudated with the new parameters of the Targets networks. This process is repeated periodically until the training process is finished. 

To avoid a correlation in the sequence used to train the agent, we implemented experience replay where we store the **`(State, Action, Reward, Next_State)`** tuples in a replay buffer and sample from it a random sequence. Therefore, the DDPG Learning with Experience Replay algorithm follows the following pseudo-code:

<p align="center">
    <img src="https://user-images.githubusercontent.com/27258035/83956363-87ae6680-a85d-11ea-8750-5a4549f24c8a.png" width="400"/>
</p>

We use the Ornstein-Uhlenbeck process to add some variance to the actions of the agents. As in this implementation has `20` agents acting in the environment, the Ornstein-Uhlenbeck class was modified so the output of sampling from it has a size of (`num of agents`,`action size`).

In this implementation, we trained `10` times after every `20` time steps with a batch size of `128` expiriences. Finally we continued training the agent until the average reward of `100` continues episode was greater than `30` or until we reach `2000` episodes.

### Neuronal Network

As explained before DDPG is an Actor-Critic method, therefore we had to define two different Neuronal Networks for both the actor and the critic.

#### Actor

For the actor we decided to implement a fully connected network with `2` hidden layers of `128` units each one and a batch normalization layer after the first hidden layer. As activation, we use the Relu function after the batch normalization layer and after the second hidden layer. Finally the tanh function was selected as activation function of the output layer.

#### Critic

For the actor we decided to implement a fully connected network with `2` hidden layers and a batch normalization layer after the one. The first layer was implemented with `128` units, and the second one with `128` + `4` (action size) units. As activation, we use the Relu function after the batch normalization layer and after the second hidden layer.

### Hyperparameters

The agent is trained with the following hyperparameters:

- **`Batch size`**  =  128
- **`Gamma`**  =  0.99
- **`TAU`**  =  1e-3
- **`Actor Learning Rate`**  =  1e-3
- **`Critic Learning Rate`**  =  1e-3
- **`Weight decay`**  =  0
- **`Steps per update`**  =  20
- **`training steps`**  =  10
- **`Noise Epsilon`**  = 1.0
- **`Epsilon Decay`**  = 1e-6


## Result

The following figure shows the reward obtained in each episode while training the agent.

<p align="center">
    <img src="https://user-images.githubusercontent.com/27258035/83963429-67ef6080-a8a6-11ea-92d9-bc2acdee9503.PNG" width="400"/>
</p>

As it can be seen, the agent reached the end criteria after **`220`**  episodes, with a 100 episode average reward of **`30.04`**.

## Improvements

To improve this implementation, we should try to use a Generalized Advantage Estimiation (e.g Lambda Return), as in the current state of this algorithm we only use a TD bootstraping of one step.
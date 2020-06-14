# Udacity's Deep Reinforcement Learnig - Project 2: Continous Control  Report

This is the report for the third project in the Udacity's Deep Reinforcement Learning Nanodegree Program. This reports explains the details of the algorithm usen to train an agent to solve play the a given scenario.

## Collaboration and Competition

The given task was to program `2` smart agents that has to move a racket to maximize the amount of times that they bounce a ball over a net, without making it fall to the ground or bouncing it out of the field of play. In this scenario this two smart agents are playing against each other, so they have to cooperate so they can keep playing. This section discusses the details of the environment where the agents are trained.

### Environment

The environment consisted in a tennis field with `2` rackets on each side of the field. The task of the smart agents is to move the rackets to bounce the ball over the net as much as possible. The enviroment was developed in Unity using the Machine Learning Agent plugin. This plugin allows developers to create enviroments where they can use to train agents using Reinforcement Learning using it's python API. For this environment each agent receive reward of `+0.1` each time they hit the ball over the net and a reward of `-0.01` if they let the ball hit the ground or hits the ball out of bound. 

### Action Space

Each agent of the environment contents a 2 dimensional continous action space corresponding to the moviment of the racket and jumping of the racket. In this case, the range of each action space is between `-1` and `1`. As this two agents are playing against each other, each of them has a coordinate system where moving towards the net correspond to a positive value of the action space, and a negative value moving away from the net.

### State Space

Each agent of the environment has a continous `8` dimensional state space corresponding the position and velocity of the ball and the racket of the agent. As this environment consist of `2` agents, the state space is a matrix of (`2`,`8`)

## Method

This section discuss the implemented algorithm to train the smart agent.

### Multiple Agent DDPG Learning with Expirience Replay

To train the agent, we implemente the Deep Deterministic Policy Gradient (DDPG) learning technique devolped by Lillicrap *et al* [paper](https://arxiv.org/abs/1509.02971)

This algorithm is a actor-critic method where we train a `2` predefined neuronal networks (actor and critic) to act according to the current states and solved a task. In these methods the critic is in charge to estimate the value function of the given state that is used by the actor to learn and predict the best action. As in DQN, we will have two identical neuronal networks to the actor and the critic that will calculate the targets that agent will learn.

This Actor-Critic Target Networks will generate the target using the last weights of the Local Networks for a period of time. After this period has finished, the Targets weights are updated with the weights of the local networks in that time step, and then the learning process of the local networks is reanudated with the new parameters of the Targets networks. This process is repeated periodically until the training process is finished. 

To avoid a correlation in the sequence used to train the agent, we implemented experience replay where we store the **`(State, Action, Reward, Next_State)`** tuples in a replay buffer and sample from it a random sequence. Therefore, the DDPG Learning with Experience Replay algorithm follows the following pseudo-code:

<p align="center">
    <img src="https://user-images.githubusercontent.com/27258035/83956363-87ae6680-a85d-11ea-8750-5a4549f24c8a.png" width="400"/>
</p>

In difference to DDPG, in this case we have `2` different agents that are playing against each other, therefore the Neuronal Network has to learn using the experience of each agent playing against each other. Each agent add a individual experience to the training. In this implementation, we trained one Neuronal Network with the experience of each agent, therefore, we used the same model to act individually for each agent.

We use the Ornstein-Uhlenbeck process to add some variance to the actions of the agents. In this implementation we had to scale the range of the noise, so we can converge the model to an optimal policy.

In this implementation, we trained `1` times after every time step with a batch size of `128` expiriences. Finally we continued training the agent until the average reward of `100` continues episode was greater than `0.5` and we reach at least the `1500` episode.

### Neuronal Network

As explained before DDPG is an Actor-Critic method, therefore we had to define two different Neuronal Networks for both the actor and the critic.

#### Actor

For the actor we decided to implement a fully connected network with `2` hidden layers of `256` units. As activation, we use the Relu function after the batch normalization layer and after the second hidden layer. Finally the tanh function was selected as activation function of the output layer.

#### Critic

For the actor we decided to implement a fully connected network with `2` hidden layers. The first layer was implemented with `256` units, and the second one with `256` + `2` (action size) units. As activation, we use the Relu function after after the second hidden layer.

### Hyperparameters

The agent is trained with the following hyperparameters:

- **`Batch size`**  =  256
- **`Gamma`**  =  0.99
- **`TAU`**  =  1e-3
- **`Actor Learning Rate`**  =  4e-4
- **`Critic Learning Rate`**  =  2e-4
- **`Weight decay`**  =  0
- **`Steps per update`**  =  1
- **`training steps`**  =  1
- **`OU Noise mu`**  =  0
- **`OU Noise Theta`**  =  0.015
- **`OU Noise Sigma`**  =  0.02

## Result

The following figure shows the reward obtained in each episode while training the agent.

<p align="center">
    <img src="https://user-images.githubusercontent.com/27258035/84601572-7bd82b00-ae81-11ea-875d-dfbdbf7ad99a.PNG" width="400"/>
</p>

As it can be seen, the agent reached the end criteria after **`987`** episodes, with a 100 episode average reward of **`0.51`**. After **`1500`** episodes we obtained a with a 100 episode average reward of **`1.93`**.

## Improvements

To improve this implementation, we should try to use a Generalized Advantage Estimiation (e.g Lambda Return), as in the current state of this algorithm we only use a TD bootstraping of one step. We could also try to change the reward so they can be trained as cooperative, so they play to maximize the reward obtained by both of them.
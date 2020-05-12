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

In this algorithm we train a predefined neuronal network to predict the value functions from a give state called Local Network. As in bellman equation the actions to estimate the value fuctions of a given state, we need to compute the value functions of the next state. To this end the DeeqQ algorithm estimates the next state value fuction with an identical neuronal network to the local one, called the Target Network. Therefore the update of the local neuronal networks weights is defined as:


This Target Network generate the next state value fuctions using the last weights of the local one in the training process for a period of time. After this period the Target weights are updated to the Local ones, and we continue training the Local network. This process is repeated for until the training process end. 


To avoid a correlation in the sequence used to train the agent, we implemented experience replay where we store the <img src="http://www.sciweavers.org/tex2img.php?eq=%28S%2CA%2CR%2CS%27%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="(S,A,R,S')" width="93" height="19" /> tuples in a replay buffer and sample from it a random sequence. Therefore, the DeepQ Learning with Expirience Replay algorithm follows the following pseudo-code:


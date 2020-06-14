from unityagents import UnityEnvironment
import numpy as np
import time
from ddpg_agent1 import Agent
from collections import deque
import torch
from itertools import count
import matplotlib.pyplot as plt

def ddpg(env,agent, n_episodes=1000, print_every=100):
    """Deep Deterministic Policy Gradient (DDPG)
    
    Params
    ======
        n_episodes  (int): maximum number of training episodes
        print_every (int): How many episodes pass between each print.
    """
    # Queue to store the last 100 max scores, the then plot a moving average.
    brain_name = env.brain_names[0]
    scores_deque = deque(maxlen=100)
    # Full score list to return and later plot.
    scores = []
    # List of the mivong avergare score over 100 episodes, to plot later.
    moving_avg_score = []
    # Define a finish flag for the learning task, a best score tracker, 
    finished_learning = False
    solved_in = -1
    
    for i_episode in range(1, n_episodes+1):        
        # Restart the Environment in train mode 
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        # Get the initial state
        states = env_info.vector_observations
        rr_state = states[0]                                    # Separate right racket state
        lr_state = states[1]                                    # Separate Left racket state
        # Reset the agents
        agent.reset()
        # Set the score of each agent to 0
        score = np.zeros(num_agents)

        # Start a timer to see how long the episode took:
        t_start = time.time()
            
        while(True):
            ## Select Actions per racket.
            rr_action = agent.act(rr_state)                    # get the action of the left racket.   
            lr_action = agent.act(lr_state)                    # get the action of the right racket.
            # Join the actions into a single vector to pass to the environment
            actions = np.vstack([rr_action, lr_action])
            
            # Step the Environment and gather all the information of the timestep
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            
            # Separate the state for the Right and Left Racket.
            next_states = env_info.vector_observations         # get next state (for each agent)
            rr_next_state = next_states[0]                     # Separate right racket state
            lr_next_state = next_states[1]                     # Separate Left racket state
        
            # Separate rewards by racket 
            rewards = env_info.rewards                         # get reward (for each agent)
            rr_reward = rewards[0]
            lr_reward = rewards[1]
            # Separate dones by racket
            dones = env_info.local_done                        # see if episode finished
            rr_done = dones[0]
            lr_done = dones[1]
       
            # Add the experience of each racket separate of each other
            agent.step(rr_state, rr_action, rr_reward, rr_next_state, rr_done)
            agent.step(lr_state, lr_action, lr_reward, lr_next_state, lr_done)
            
            # roll over states to next time step
            rr_state = rr_next_state                           
            lr_state = lr_next_state
            
            # update the score (for each agent)
            score += rewards                                   

            # Check if the Episode is over
            if np.any(dones):
                break
         
        # Stop the episode timer
        episode_timer = time.time() - t_start
        
        # Save the max of the two scores between the two rackets.
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        # Also save the current moving average score
        moving_avg_score.append(np.mean(scores_deque))
        
        # Print the current status of the training
        print('\rEpisode: {} ({:4.2f}s)\tScore: {:4.2f}\tMoving Average Score: {:4.2f}       '.format(i_episode, episode_timer, scores[-1], np.mean(scores_deque)), end='')
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode: {} ({:4.2f}s)\tScore: {:4.2f}\tMoving Average Score: {:4.2f}       '.format(i_episode, episode_timer, scores[-1], np.mean(scores_deque)))
            
        # Save the code if the training already succeded    
        if np.mean(scores_deque)>=0.5 and finished_learning == False:
            print('\nEnvironment solved in {:d} episodes! Max Average Score over 100 episodes: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_SUCCESS.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_SUCCESS.pth')
            finished_learning = True
            # Save how many episodes it took to solve the task
            solved_in = i_episode
            
        # Check if the current average is the best of all the averages, and keep updating the  "Success weights" the current best candidate
        if len(moving_avg_score) > 2:
            if moving_avg_score[-1] > np.max(moving_avg_score[:-1]) and moving_avg_score[-1] > 0.5:
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_SUCCESS.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_SUCCESS.pth')
            
    
    ## Print the final results of the training:
    print("\n\nTraining finished:")
    print("\t[+] Total nuber of episodes:    {}".format(n_episodes))
    print("\t[+] Best 100-average score:     {:4.2f}".format(np.max(moving_avg_score)))
    print("\t[+] Best individual game score: {:4.2f}".format(np.max(scores)))
    
    # Check if the task was actually solved
    if solved_in != -1:
        print("Task solved in {} Episodes!".format(solved_in))
    
    
    return scores, moving_avg_score



def maddpg(env, agent, n_episodes=2000, summary_freq=10):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    brain_name = env.brain_names[0]
    
    episode_scores = []                 # list containing scores from each episode
    average_scores = []                 # list containing the 100th average window scores
    scores_window = deque(maxlen=100)   # last 100 scores
    start_time = time.time()            # init time
    # -------------------------- Training ------------------------------------#
    for i_episode in range(1, n_episodes+1):
        timer = time.time()
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        
        player1_state = env_info.vector_observations[0]
        player2_state = env_info.vector_observations[1]
        
        agent.reset()
        scores = np.zeros(num_agents)
        a = 0
        while True:
            print("\r {}".format(a),end="")
            a = a + 1
            player1_action = agent.act(player1_state)
            player2_action = agent.act(player2_state)
            
            actions = np.vstack([player1_action, player2_action])
            env_info = env.step(actions)[brain_name]
            
            player1_next_state = env_info.vector_observations[0]
            player2_next_state = env_info.vector_observations[1]
            
            rewards = env_info.rewards
            player1_reward = env_info.rewards[0]
            player2_reward = env_info.rewards[1]
            
            dones = env_info.local_done
            player1_done = env_info.local_done[0]
            player2_done = env_info.local_done[1]

            agent.step(player1_state, player1_action, player1_reward, player1_next_state, player1_done)
            agent.step(player2_state, player2_action, player2_reward, player2_next_state, player2_done)

            player1_state = player1_next_state
            player2_state = player2_next_state

            scores += rewards

            if np.any(dones):
                timer = time.time()-timer
                break
        
        episode_score = np.max(scores)                   # calculate episode score
        scores_window.append(episode_score)               # save most recent score
        episode_scores.append(episode_score)              # save most recent score
        average_scores.append(np.mean(scores_window))     # save current average score

        # ------------------------ report and checkpoint saaving ----------------#
        training_time =  time.time()-start_time
        hours = int(training_time // 3600)
        minutes = int(training_time % 3600 // 60)
        seconds = training_time % 60
        
        if i_episode % summary_freq == 0:
            print('***\033[1mEpisode {}\tAverage Score: {:.2f}\t Max Score: {:.2f}\tTime training: {:d}:{:d}:{:05.2f}***\033[0m'.format(i_episode, 
                                                                                    average_scores[-1],
                                                                                    np.max(average_scores),
                                                                                    hours,
                                                                                    minutes,
                                                                                    seconds))
        if  np.mean(scores_window) >= 0.5:
            print('\n\033[1m\033[92mEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\t Max Score: {:.2f}\tTraining time: {:d}:{:d}:{:05.2f}\033[0m'.format(i_episode,
                                                                                                          average_scores[-1],
                                                                                                          np.max(average_scores),
                                                                                                          hours,                                                                                                                                                                    hours,
                                                                                                          minutes,
                                                                                                          seconds))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            
        if average_scores[-1]>=0.5 and np.max(average_scores[:-1]) <= average_scores[-1] and np.mean(scores_window) >= 0.5:
            torch.save(agent.actor_local.state_dict(), 'actor_best.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_best.pth')
            break
    return episode_scores, average_scores



def main():
    env = UnityEnvironment(file_name='Tennis_Windows_x86_64/Tennis.exe')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # Instantiate agent
    agent = Agent(state_size=24, action_size=2, random_seed=1)
    
    # Train Agent
    scores, average_scores = ddpg(env,agent)
    env.close()


if __name__ == "__main__":
    main()

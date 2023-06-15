# from TD3 import TD3Agent, ReplayBuffer
# import gym
# import numpy as np
# import os


# import os

# # Set the directory for saving checkpoints and results
# checkpoint_dir = "checkpoints"
# result_dir = "results"

# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)


# # Create your Gym environment
# env = gym.make('BipedalWalker-v3')

# # Set the state and action dimensions
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# max_action = float(env.action_space.high[0])

# # Create an instance of the TD3 agent
# td3_agent = TD3Agent(state_dim, action_dim, max_action)

# # Set hyperparameters
# max_episodes = 250_000
# max_timesteps = 200
# batch_size = 128

# # Create the replay buffer
# buffer_size = 100_000
# replay_buffer = ReplayBuffer(buffer_size)

# # Training loop
# start_episode = 100_000

# td3_agent.load_models(checkpoint_dir, start_episode)
# for episode in range(start_episode, max_episodes):
#     state = env.reset()
#     episode_reward = 0
#     done = False
    
#     env.render()
#     for t in range(max_timesteps):
#         # Select an action with exploration
#         action = td3_agent.get_action(state)
#         action = np.clip(action, env.action_space.low, env.action_space.high)

#         # Execute the action in the environment
#         next_state, reward, done, _ = env.step(action)

#         # Store the transition in the replay buffer
#         replay_buffer.add(state, action, reward, next_state, done)

#         state = next_state
#         episode_reward += reward

#         if done:
#             break
        

    
#     # Train the agent
#     if len(replay_buffer.buffer) > batch_size:
#         td3_agent.train(batch_size)

#     # Print the episode rewards
#     print("Episode:", episode, "Reward:", episode_reward)


#     # Save model checkpoints
#     if episode % 10000 == 0:
#         # checkpoint_path = os.path.join(checkpoint_dir, f"episode_{episode}.h5")
#         td3_agent.save_models(checkpoint_dir, episode)

#     # Save episode rewards
#     with open(os.path.join(result_dir, "episode_rewards.txt"), "a") as f:
#         f.write(f"{episode},{episode_reward}\n")

        

# # Evaluate the trained agent
# eval_episodes = 10
# eval_rewards = []

# for _ in range(eval_episodes):
#     state = env.reset()
#     episode_reward = 0
#     done = False

#     while not done:
#         action = td3_agent.get_action(state)
#         action = np.clip(action, env.action_space.low, env.action_space.high)

#         next_state, reward, done, _ = env.step(action)
#         state = next_state
#         episode_reward += reward

#     eval_rewards.append(episode_reward)

# # Print average evaluation rewards
# print("Average Evaluation Reward:", np.mean(eval_rewards))

import gym
import numpy as np
from TD3 import TD3Agent
import os

result_dir = 'results'
if __name__ == '__main__':
    #env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('BipedalWalker-v3')
    env = gym.make("BipedalWalker-v3")
    # env = gym.make('Pendulum-v0')
    # env.render()
    agent = TD3Agent(input_dims=env.observation_space.shape, env = env, batch_size=100)

    n_games = 1000
    # filename = 'plots/' + 'walker_' + str(n_games) + '_games.png'

    best_score = env.reward_range[0]
    score_history = []

    # agent.load_models(checkpoint_dir='pendulum2/')

    for i in range(n_games):
        
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.train()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models(checkpoint_dir='bipedal')

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

            # Save episode rewards
        with open(os.path.join(result_dir, "bipedal1_results.txt"), "a") as f:
            f.write(f"{i},{score}, {avg_score}\n")

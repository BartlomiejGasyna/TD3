import gym
import numpy as np
from TD3 import TD3Agent
import os

result_dir = 'results'
if __name__ == '__main__':
    env = gym.make('Pendulum-v0')

    agent = TD3Agent(input_dims=env.observation_space.shape, env = env, batch_size=100)

    n_games = 1000

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

        # zapis najlepszych wag - średnia 100 epizodów
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models(checkpoint_dir='bipedal')

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

        # zapis wyników do pliku txt - w celu stworzenia wykres
        with open(os.path.join(result_dir, "bipedal1_results.txt"), "a") as f:
            f.write(f"{i},{score}, {avg_score}\n")

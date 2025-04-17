from a2c_dsa import reset_env, get_action, my_step, store_transition, preprocess_obs, compute_a2c_loss
from utils.env import CogSatDSAEnv
def train_multiple_episodes(n_episodes=100):

    for episode in range(n_episodes):
        step_count = 0  # Reset step count for each episode
        obs = reset_env()
        done = False
        episode_reward = 0
        while not done:
            action = get_action(obs)
            next_obs, reward, done = my_step(action)
            store_transition(reward, done, next_obs)
            obs = next_obs
            step_count += 1
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Total Steps = {step_count}")
            episode_reward += reward
            if done:
                print("+++++++++++Episode finished+++++++++++++")
                break  # Optional, since the loop exits on `done` anyway
        print(f"Episode {episode + 1} finished with total reward: {episode_reward:.2f}")


train_multiple_episodes(10)
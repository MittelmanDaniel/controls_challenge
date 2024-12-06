import os
import numpy as np
from typing import List
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from car_env import CarEnv

def make_env(data_path: str, model_path: str):
    """
    Environment factory function for vectorized environments.
    
    Args:
        data_path: Path to the data file
        model_path: Path to the physics model
    """
    def _init():
        return CarEnv(data_path, model_path, debug=False)
    return _init

def evaluate_model(model, vec_env, num_episodes: int = 10) -> List[float]:
    """
    Evaluate a trained RL model over multiple episodes.
    
    Args:
        model: Trained RL model (e.g., PPO)
        vec_env: Vectorized environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        List of rewards for each completed episode
    """
    # Initialize tracking variables
    episode_rewards = np.zeros(vec_env.num_envs)
    completed_episodes_rewards = []
    num_completed_episodes = 0
    
    # Reset all environments
    obs = vec_env.reset()  # Note: Now properly unpacking reset return values
    
    while num_completed_episodes < num_episodes:
        # Get actions from the model
        actions, _ = model.predict(obs, deterministic=True)
        
        # Step the environments (SubprocVecEnv combines terminated/truncated into dones)
        obs, rewards, dones, infos = vec_env.step(actions)
        
        # Update episode rewards
        episode_rewards += rewards
        
        # Check for completed episodes
        for env_idx in range(vec_env.num_envs):
            if dones[env_idx]:
                completed_episodes_rewards.append(float(episode_rewards[env_idx]))
                episode_rewards[env_idx] = 0
                num_completed_episodes += 1
                
                if num_completed_episodes >= num_episodes:
                    break
    
    # Calculate statistics
    mean_reward = np.mean(completed_episodes_rewards)
    std_reward = np.std(completed_episodes_rewards)
    
    print(f"\nEvaluation Results:")
    print(f"Number of episodes: {num_episodes}")
    print(f"Mean episode reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min episode reward: {min(completed_episodes_rewards):.2f}")
    print(f"Max episode reward: {max(completed_episodes_rewards):.2f}")
    
    return completed_episodes_rewards

def main():
    # Configuration
    model_path = "./models/tinyphysics.onnx"
    data_path = "./data/00000.csv"
    n_envs = 16
    total_timesteps = 16 * 200000
    
    # Create vectorized environments
    envs = [make_env(data_path, model_path) for _ in range(n_envs)]
    vec_env = SubprocVecEnv(envs)
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
    )
    
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed!")
    
    # Save the trained model
    save_path = "trained_model"
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Evaluate the trained model
    print("\nStarting evaluation...")
    episode_rewards = evaluate_model(model, vec_env, num_episodes=10)
    
    # Clean up
    vec_env.close()

if __name__ == "__main__":
    main()
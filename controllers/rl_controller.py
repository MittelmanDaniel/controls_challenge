from . import BaseController
import numpy as np
from stable_baselines3 import PPO

class Controller(BaseController):
  """
  A controller that uses a pretrained RL policy (e.g. PPO) to determine actions.
  
  The RL policy expects the observation in the same format used during training:
  obs = [current_lataccel, target_lataccel, roll_lataccel, v_ego, a_ego]
  
  Make sure to place 'rl_policy.zip' in a known path and adjust the path below.
  """
  def __init__(self, model_path="./rl_policy.zip"):
    # Load the trained RL model
    self.model = PPO.load(model_path)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # Construct observation similar to the CarEnv
    obs = np.array([
      current_lataccel,
      target_lataccel,
      state.roll_lataccel,
      state.v_ego,
      state.a_ego
    ], dtype=np.float32)
    
    # Get action from the RL model
    # deterministic=True for stable actions, set to False for exploratory
    action, _ = self.model.predict(obs, deterministic=True)
    return float(action[0])

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, STEER_RANGE
from controllers import BaseController

class DummyController(BaseController):
    def update(self, *args, **kwargs):
        return 0.0

class CarEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, data_path, model_path, debug=False):
        super().__init__()
        
        self.model = TinyPhysicsModel(model_path, debug=debug)
        self.sim = TinyPhysicsSimulator(self.model, str(data_path), controller=DummyController(), debug=debug)
        self.sim.reset()

        # Observation: [current_lataccel, target_lataccel, roll_lataccel, v_ego, a_ego]
        obs_low = np.array([-5, -5, -10, 0.0, -5.0], dtype=np.float32)
        obs_high = np.array([5, 5, 10, 50.0, 5.0], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([STEER_RANGE[0]], dtype=np.float32),
                                       high=np.array([STEER_RANGE[1]], dtype=np.float32),
                                       shape=(1,),
                                       dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        obs = self.sim.get_observation()
        info = {}
        return obs, info

    def step(self, action):
        self.sim.apply_action(float(action[0]))

        obs = self.sim.get_observation()
        reward = self.sim.get_reward()
        terminated = self.sim.is_done()
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

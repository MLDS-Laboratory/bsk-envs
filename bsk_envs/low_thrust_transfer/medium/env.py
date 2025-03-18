import gymnasium as gym
import numpy as np
from numpy.linalg import norm
from gymnasium import spaces
from Basilisk.architecture import astroConstants
from bsk_envs.low_thrust_transfer.medium.sim import LowThrustTransfer3DOFSimulator


RE = 149.78e6 * 1000
VE = np.sqrt(astroConstants.MU_SUN / (RE / 1000)) * 1000
RM = 228e6 * 1000
TMAX = 0.5
UEQ = 19.6133 * 1000
TF = 358.79

class LowThrustTransfer3DOFEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, m0=1000, max_steps=40, render_mode=None):
        super(LowThrustTransfer3DOFEnv, self).__init__()
        self.m0 = m0
        self.mk = m0
        self.mpk = 0
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.step_count = 0
        self.obs = None
        self.simulator = None

        self.observation_space = spaces.Box(low=-1e16, high=1e16, shape=(7,))
        self.action_space = spaces.Box(-1, 1, shape=(3,))

    def max_action(self):
        return (TMAX / self.mk) * (TF * 86400 / self.max_steps)
    
    def tsiolkowsky(self, dv):
        mk1 = self.mk * np.exp(-np.linalg.norm(dv) / UEQ)
        return mk1

    def _get_state(self):
        r = self.obs['r_S_N'] / RE
        v = self.obs['v_S_N'] / VE
        m = [self.mk / self.m0]
        return np.concatenate((r, v, m), dtype=np.float32) 
    
    def _get_reward(self, action):
        reward = -self.mpk / self.m0
        reward -= 10 * max(0, norm(action) - 1)
        if self.step_count == self.max_steps:
            dv = self.obs['v_S_N'] - self.obs['v_M_N']
            reward -= (self.mk - self.tsiolkowsky(dv)) / self.m0
            r_penalty = norm(self.obs['r_S_N'] - self.obs['r_M_N']) / norm(self.obs['r_M_N'])
            v_penalty = max(0, (norm(dv) - self.max_action()) / norm(self.obs['v_M_N']))
            reward -= 5 * max(0, max(r_penalty, v_penalty) - 1e-2)
        return reward

    def _get_terminal(self):
        return self.step_count == self.max_steps
        
    def reset(self, seed=None, options={}):
        if self.simulator is not None:
            del self.simulator
        self.simulator = LowThrustTransfer3DOFSimulator(
            max_steps=self.max_steps,
            render_mode=self.render_mode
        )
        self.obs = self.simulator.init()
        self.mk = self.m0
        self.mpk = 0.0
        self.step_count = 0
        state = self._get_state()
        return state, {}
    
    def step(self, action):
        dv = self.max_action() * action
        mk1 = self.tsiolkowsky(dv)
        self.mpk = self.mk - mk1
        self.mk = mk1

        self.obs = self.simulator.run(dv)
        self.step_count += 1
        next_state = self._get_state()
        reward = self._get_reward(action)
        done = self._get_terminal()

        return next_state, reward, done, False, {}
    
    def close(self):
        if self.simulator is not None:
            del self.simulator
    
    

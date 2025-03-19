import gymnasium as gym
import numpy as np
from numpy.linalg import norm
from gymnasium import spaces
from Basilisk.architecture import astroConstants
from bsk_envs.low_thrust_transfer.medium.sim import LowThrustTransfer3DOFSimulator


RE = 149.78e6 * 1000
VE = np.sqrt(astroConstants.MU_SUN / (RE / 1000)) * 1000

class LowThrustTransfer3DOFEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(
            self, 
            m_init=1000, 
            u_eq=19.6133 * 1000, 
            T_max=0.5, 
            tf=358.79 * 86400,
            offset=0,
            sigma=0,  
            max_steps=40, 
            render_mode=None
        ):
        super(LowThrustTransfer3DOFEnv, self).__init__()
        self.m_init = m_init # initial mass in kg
        self.m = m_init # current mass in kg
        self.u_eq = u_eq # exhaust velocity in m/s
        self.T_max = T_max # maximum thrust in N
        self.tf = tf # total time in seconds
        self.m_prop = 0 # propellent consumed in kg
        self.offset = np.radians(offset)
        self.rot = np.array([
            [np.cos(self.offset), 0, np.sin(self.offset)],
            [0, 1, 0],
            [-np.sin(self.offset), 0, np.cos(self.offset)]
        ])
        self.sigma = sigma
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.step_count = 0
        self.obs = None
        self.simulator = None

        self.observation_space = spaces.Box(low=-1e16, high=1e16, shape=(7,))
        self.action_space = spaces.Box(-1, 1, shape=(3,))

    def max_action(self):
        return (self.T_max / self.m) * (self.tf / self.max_steps)
    
    def tsiolkowsky(self, dv):
        m = self.m * np.exp(-np.linalg.norm(dv) / self.u_eq)
        return m

    def _get_state(self):
        r = self.obs['r_S_N'] / RE
        v = self.obs['v_S_N'] / VE
        m = [self.m / self.m_init]
        return np.concatenate((r, v, m), dtype=np.float32) 
    
    def _get_reward(self, action):
        reward = -self.m_prop / self.m_init
        reward -= 100 * max(0, norm(action) - 1)
        if self.step_count == self.max_steps:
            r_penalty = norm(self.obs['r_S_N'] - self.obs['r_M_N']) / norm(self.obs['r_M_N'])
            v_penalty = norm(self.obs['v_S_N'] - self.obs['v_M_N']) / norm(self.obs['v_M_N'])
            reward -= 50 * max(0, max(r_penalty, v_penalty) - 1e-3)
        return reward

    def _get_terminal(self):
        return self.step_count == self.max_steps
        
    def reset(self, seed=None, options={}):
        if self.simulator is not None:
            del self.simulator
        self.simulator = LowThrustTransfer3DOFSimulator(
            tf=self.tf,
            max_steps=self.max_steps,
            render_mode=self.render_mode
        )
        self.obs = self.simulator.init()
        self.m = self.m_init
        self.m_prop = 0.0
        self.step_count = 0
        state = self._get_state()
        return state, {}
    
    def step(self, action):
        dv = self.max_action() * action
        dv = self.rot @ dv
        dv += np.random.normal(loc=0, scale=self.sigma)
        m = self.tsiolkowsky(dv)
        self.m_prop = self.m - m
        self.m = m

        self.obs = self.simulator.run(dv)
        self.step_count += 1
        next_state = self._get_state()
        reward = self._get_reward(action)
        done = self._get_terminal()
        return next_state, reward, done, False, {}
    
    def close(self):
        if self.simulator is not None:
            del self.simulator
    
    

from gymnasium.envs.registration import register

register(
    id='LowThrustTransfer3DOF-v0', 
    entry_point='bsk_envs.low_thrust_transfer.medium.env:LowThrustTransfer3DOFEnv'
)
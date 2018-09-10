from gym.envs.registration import register

register(
    id='CoDrone-v0',
    entry_point='gym_codrone.envs:CoDroneEnv',
    timestep_limit=100,
)

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CoDrone-v0',
    entry_point='gym-codrone.envs:CoDroneEnv',
    timestep_limit=100,
    
)


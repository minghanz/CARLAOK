from gym.envs.registration import register
# Routing
# ----------------------------------------
register(
    id='carla-v0',
    entry_point='gym_routing.envs:CarlaEnv'
)

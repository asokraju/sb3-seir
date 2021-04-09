from gym.envs.registration import register

register(
    id='seir-v0',
    entry_point='gym_seir.envs:SeirEnv',
)
register(
    id='tseir-v0',
    entry_point='gym_seir.envs:TSeirEnv',
)
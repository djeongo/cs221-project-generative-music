from gym.envs.registration import register

register(
        id='melody-v0',
        entry_point='gym_melody.envs:MelodyEnv',
        )

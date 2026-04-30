from gymnasium.envs.registration import register

register(
    id='TwentyFortyEight-v0',
    entry_point='twenty_forty_eight.envs:TwentyFortyEightEnv',
)

from gym.envs.registration import register

register(
    id='BaseEnv-v0', 						# Modify here
    entry_point='CustomEnv.envs:BaseEnv', 	# Modify here
)

# New environments should be registered after here
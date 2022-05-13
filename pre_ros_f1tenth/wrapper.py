from pre_ros_f1tenth.f1tenth_gym.f110_env import F110Env



class WrappedEnv(F110Env):
    def __init__(self):
        super().__init__()
    
    def step(self, action):
        pass 

    def reset(self):
        pass


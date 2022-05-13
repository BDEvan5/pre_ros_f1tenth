from pre_ros_f1tenth.f1tenth_gym.f110_env import F110Env

from safety_system_ros.Planners.PurePursuitPlanner import PurePursuitPlanner


conf = load_config('config_file')
env = F110Env(map_name='columbia_small', map_ext='.png')
planner = PurePursuitPlanner(conf, 'columbia_small')

env.reset()
for i in range(100):
    planner.plan()
    








from pre_ros_f1tenth.f1tenth_gym.f110_env import F110Env
from pre_ros_f1tenth.TrainTest import evaluate_vehicle
from pre_ros_f1tenth.utils import load_conf
from pre_ros_f1tenth.SimWrapper import PreRosSim

from safety_system_ros.Planners.PurePursuitPlanner import PurePursuitPlanner


def main():
    sim = PreRosSim()
    sim.run_test()



if __name__ == '__main__':
    main()





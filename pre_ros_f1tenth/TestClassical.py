from pre_ros_f1tenth.TestWrapper import BaseWrapper
from safety_system_ros.Planners.PurePursuitPlanner import PurePursuitPlanner
from pre_ros_f1tenth.utils import *


class ClassicalTest(BaseWrapper):
    def __init__(self, testing_params: str):
        super().__init__(testing_params)
        self.test_params = load_conf(testing_params)

        self.planner = PurePursuitPlanner(self.conf, self.test_params.map_name)

        self.supervision = False

        self.run_testing()


if __name__ == "__main__":
    testing_params = "testing_params"
    ClassicalTest(testing_params)


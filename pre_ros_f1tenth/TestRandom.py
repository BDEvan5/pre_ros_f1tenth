from pre_ros_f1tenth.TestWrapper import BaseWrapper
from safety_system_ros.Planners.RandomPlanner import RandomPlanner
from pre_ros_f1tenth.utils import *
from safety_system_ros.Supervisor import Supervisor, LearningSupervisor


class ClassicalTest(BaseWrapper):
    def __init__(self, testing_params: str):
        super().__init__(testing_params)
        self.test_params = load_conf(testing_params)

        self.planner = RandomPlanner(self.conf)

        self.supervision = True
        self.supervisor = Supervisor(self.conf, self.test_params.map_name)

        self.run_testing()


if __name__ == "__main__":
    testing_params = "testing_params"
    ClassicalTest(testing_params)


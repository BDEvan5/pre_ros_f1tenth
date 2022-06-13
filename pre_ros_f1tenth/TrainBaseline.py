
from pre_ros_f1tenth.utils import *
from pre_ros_f1tenth.TestWrapper import BaseWrapper

from safety_system_ros.Planners.TrainingAgent import TestVehicle, TrainVehicle
from safety_system_ros.Supervisor import Supervisor, LearningSupervisor


import numpy as np
import time
from safety_system_ros.utils.Reward import RaceTrack, DistanceReward




class TrainSimulation(BaseWrapper):
    def __init__(self, test_params):
        super().__init__(test_params)

        #train  
        self.race_track = RaceTrack(self.test_params.map_name)
        self.race_track.load_center_pts()
        self.reward = DistanceReward(self.race_track)

        self.n_train_steps = self.test_params.n_train_steps
        self.previous_observation = None

    def run_training_evaluation(self):
        self.planner = TrainVehicle(self.conf, self.test_params.agent_name)
        self.supervision = False
        self.completed_laps = 0

        self.run_training()

        self.planner = TestVehicle(self.conf, self.test_params.agent_name)

        self.n_test_laps = self.test_params.n_test_laps

        self.lap_times = []
        self.completed_laps = 0

        self.run_testing()

    def retest_vehicle(self):
        self.planner = TestVehicle(self.conf, self.test_params.agent_name)
        self.supervision = False

        self.n_test_laps = self.test_params.n_test_laps

        self.lap_times = []
        self.completed_laps = 0

        self.run_testing()


    def run_training(self):
        start_time = time.time()
        print(f"Starting Baseline Training: {self.planner.name}")

        lap_counter, crash_counter = 0, 0
        observation = self.reset_simulation()

        for i in range(self.n_train_steps):
            self.prev_obs = observation
            action = self.planner.plan(observation)
            observation = self.run_step(action)

            self.planner.agent.train(2)

            if self.show_train: self.env.render('human_fast')

            if observation['lap_done'] or observation['colision_done'] or observation['current_laptime'] > self.conf.max_laptime:
                self.planner.done_entry(observation)

                if observation['lap_done']:
                    if self.verbose: print(f"{i}::Lap Complete {lap_counter} -> FinalR: {observation['reward']} -> LapTime {observation['current_laptime']:.2f} -> TotalReward: {self.planner.t_his.rewards[self.planner.t_his.ptr-1]:.2f}")

                    lap_counter += 1
                    self.completed_laps += 1

                elif observation['colision_done'] or self.race_track.check_done(observation):

                    if self.verbose: print(f"{i}::Lap Crashed -> FinalR: {observation['reward']} -> LapTime {observation['current_laptime']:.2f} -> TotalReward: {self.planner.t_his.rewards[self.planner.t_his.ptr-1]:.2f}")
                    crash_counter += 1

                # print(f"exp: {self.planner.agent.exploration_rate}")
                observation = self.reset_simulation()
                    
            
        self.planner.t_his.print_update(True)
        self.planner.t_his.save_csv_data()
        self.planner.agent.save(self.planner.path)

        train_time = time.time() - start_time
        print(f"Finished Training: {self.planner.name} in {train_time} seconds")
        print(f"Crashes: {crash_counter}")


        print(f"Training finished in: {time.time() - start_time}")


if __name__ == "__main__":
    test_params = "baseline_params"
    t = TrainSimulation(test_params)
    # t.run_training_evaluation()

    t.retest_vehicle()


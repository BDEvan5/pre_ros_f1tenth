from pre_ros_f1tenth.f1tenth_gym.f110_env import F110Env
from pre_ros_f1tenth.utils import *

from safety_system_ros.Planners.PurePursuitPlanner import PurePursuitPlanner
from safety_system_ros.Planners.TrainingAgent import TestVehicle, TrainVehicle
from safety_system_ros.Planners.RandomPlanner import RandomPlanner
from safety_system_ros.Supervisor import Supervisor, LearningSupervisor


import numpy as np
import time
from safety_system_ros.utils.Reward import RaceTrack, DistanceReward



class BaseWrapper():
    def __init__(self, testing_params: str):
        self.test_params = load_conf(testing_params)
        self.map_name=self.test_params.map_name
        self.conf = load_conf("config_file")

        self.env = F110Env(map=self.map_name, map_ext=".pgm")
        # self.env = F110Env(map=self.map_name, map_ext=".png")
        self.planner = None

        self.n_test_laps = self.test_params.n_test_laps
        self.lap_times = []
        self.completed_laps = 0
        self.prev_obs = None

        self.race_track = None
        self.reward = None

        # self.supervision = self.test_params.supervision
        # self.supervisor = Supervisor(self.conf, self.test_params.map_name)

        # flags 
        self.show_test = self.test_params.show_test
        self.show_train = self.test_params.show_train
        self.verbose = self.test_params.verbose

    def run_testing(self):

        start_time = time.time()

        for i in range(self.n_test_laps):
            observation = self.reset_simulation()

            while not observation['colision_done'] and not observation['lap_done']:
                action = self.planner.plan(observation)
                if self.supervision:
                    action = self.supervisor.supervise(observation['state'], action)
                observation = self.run_step(action)
                if self.show_test: self.env.render('human_fast')

            if observation['lap_done']:
                if self.verbose: print(f"Lap {i} Complete in time: {observation['current_laptime']}")
                self.lap_times.append(observation['current_laptime'])
                self.completed_laps += 1

            if observation['colision_done']:
                if self.verbose: print(f"Lap {i} Crashed in time: {observation['current_laptime']}")
                    
        print(f"Tests are finished in: {time.time() - start_time}")

        success_rate = (self.completed_laps / (self.n_test_laps) * 100)
        if len(self.lap_times) > 0:
            avg_times, std_dev = np.mean(self.lap_times), np.std(self.lap_times)
        else:
            avg_times, std_dev = 0, 0

        print(f"Crashes: {self.n_test_laps - self.completed_laps} VS Completes {self.completed_laps} --> {success_rate:.2f} %")
        print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

        eval_dict = {}
        eval_dict['success_rate'] = float(success_rate)
        eval_dict['avg_times'] = float(avg_times)
        eval_dict['std_dev'] = float(std_dev)

        run_dict = vars(self.test_params)
        run_dict.update(eval_dict)
        run_dict['vehicle_path'] = self.conf.directory + self.conf.vehicle_path

        save_conf_dict(run_dict)

    def run_step(self, action):
        sim_steps = self.conf.sim_steps

        sim_steps, done = sim_steps, False
        while sim_steps > 0 and not done:
            obs, step_reward, done, _ = self.env.step(action[None, :])
            sim_steps -= 1
        
        observation = self.build_observation(obs, done)

        return observation

    def build_observation(self, obs, done):
        """Build observation

        Returns 
            state:
                [0]: x
                [1]: y
                [2]: yaw
                [3]: v
                [4]: steering
            scan:
                Lidar scan beams 
            
        """
        observation = {}
        observation['current_laptime'] = obs['lap_times'][0]
        observation['scan'] = obs['scans'][0] 
        
        pose_x = obs['poses_x'][0]
        pose_y = obs['poses_y'][0]
        theta = obs['poses_theta'][0]
        linear_velocity = obs['linear_vels_x'][0]
        steering_angle = obs['steering_deltas'][0]
        state = np.array([pose_x, pose_y, theta, linear_velocity, steering_angle])

        observation['state'] = state
        observation['lap_done'] = False
        observation['colision_done'] = False

        observation['reward'] = 0.0
        if done and obs['lap_counts'][0] == 0: 
            observation['colision_done'] = True
        if self.race_track is not None:
            if self.race_track.check_done(observation) and obs['lap_counts'][0] == 0:
                observation['colision_done'] = True
            

        if obs['lap_counts'][0] == 1:
            observation['lap_done'] = True

        if self.reward:
            observation['reward'] = self.reward(observation, self.prev_obs)

        return observation

    def reset_simulation(self):
        reset_pose = np.zeros(3)[None, :]

        # reset_pose[0, 2] = -np.pi
        if self.map_name == 'example_map':
            reset_pose[0, 2] = np.pi/2
        obs, step_reward, done, _ = self.env.reset(reset_pose)

        if self.show_train: self.env.render('human_fast')

        observation = self.build_observation(obs, done)
        self.prev_obs = observation
        if self.race_track is not None:
            self.race_track.max_distance = 0.0

        return observation



def main():
    pass
    # sim = TestSimulation("testing_params")
    # sim.run_testing()

    # sim = TrainSimulation("testing_params")
    # sim.run_training_evaluation()

if __name__ == '__main__':
    main()




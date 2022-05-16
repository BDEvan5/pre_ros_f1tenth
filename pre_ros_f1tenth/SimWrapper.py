from pre_ros_f1tenth.f1tenth_gym.f110_env import F110Env
from pre_ros_f1tenth.utils import load_conf
from safety_system_ros.Planners.PurePursuitPlanner import PurePursuitPlanner
from safety_system_ros.Planners.TrainingAgent import TestVehicle, TrainVehicle
from safety_system_ros.Supervisor import Supervisor, LearningSupervisor

import numpy as np

class PreRosSim(F110Env):
    def __init__(self):
        self.test_params = load_conf('testing_params')
        map_name=self.test_params.map_name
        self.conf = load_conf(self.test_params.config_filename)
        super().__init__(map=map_name, map_ext=".png")

        

        self.n_laps = self.test_params.n_laps

        self.complete_laps = 0
        self.running = True
        self.current_lap_time = 0.0


    # this is an overide
    def run_step(self, action):
        sim_steps = self.conf.sim_steps

        sim_steps, done = sim_steps, False
        while sim_steps > 0 and not done:
            obs, step_reward, done, _ = self.step(action[None, :])
            sim_steps -= 1
        
        observation = self.build_observation(obs, done)

        return observation

    def build_observation(self, obs, done):
        self.current_lap_time = obs['lap_times'][0]
        observation = {}
        inds = np.arange(0, 1080, 40)
        observation["scan"] = obs['scans'][0][inds]
        
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
        if done: 
            observation['reward'] = -1.0
            observation['colision_done'] = True
        if obs['lap_counts'][0] == self.complete_laps +1:
            observation['reward'] = 1.0
            observation['lap_done'] = True

        return observation

    def reset_simulation(self):
        reset_pose = np.zeros(3)[None, :]
        obs, step_reward, done, _ = self.reset(reset_pose)

        observation = self.build_observation(obs, done)

        return observation

    

    def lap_done(self):
        print(f"Run {self.complete_laps} Complete in time: {self.current_lap_time}")
        self.lap_complete_callback()

        self.complete_laps += 1

        if self.complete_laps == self.n_laps:
            self.running = False
            # self.save_data_callback()
            # self.ego_reset()
            # self.destroy_node()

        self.current_lap_time = 0.0

    def lap_complete_callback(self):
        pass

class PreRosTesting(PreRosSim):
    def __init__(self):
        super().__init__()

        planner_dict = {'pure_pursuit': PurePursuitPlanner,
                        'random': RandomPlanner,
                        'agent': TestVehicle}

        self.planner = planner_dict[self.test_params.planner](self.conf, self.test_params)
        print(self.planner.name)

        self.supervision = self.test_params.supervision
        self.supervisor = Supervisor(self.conf, self.test_params.map_name)

    def run_testing(self):
        observation = self.reset_simulation()

        while self.running:
            action = self.planner.plan(observation)
            if self.supervision:
                action = self.supervisor.supervise(observation['state'], action)
            observation = self.run_step(action)
            self.render('human_fast')

            if observation['lap_done']:
                self.lap_done()
                
            if observation['colision_done']:
                print(f"Colission Observerd")
                break

        print(f"Tests are finished")

class PreRosTraining(PreRosSim):
    def __init__(self):
        super().__init__()


        planner_dict = {'pure_pursuit': PurePursuitPlanner,
                        'random': RandomPlanner,
                        'agent': TrainVehicle}

        self.planner = planner_dict[self.test_params.planner](self.conf, self.test_params)
        print(self.planner.name)

        self.supervision = True
        self.supervisor = LearningSupervisor(self.planner, self.conf, self.test_params.map_name)

    def run_training(self):
        observation = self.reset_simulation()

        while self.running:
            action = self.supervisor.plan(observation) 
            observation = self.run_step(action)

            self.planner.agent.train(2)
            # self.render('human_fast')

            if observation['lap_done']:
                self.lap_done()
                
            if observation['colision_done']:
                print(f"Colission Observerd")
                break

        self.save_data_callback()
        print(f"Tests are finished")

    def lap_complete_callback(self):
        print(f"Interventions: {self.supervisor.ep_interventions}")
        self.supervisor.lap_complete(self.current_lap_time)


    def save_data_callback(self):
        self.planner.agent.save(self.planner.path)
        self.planner.t_his.print_update(True)
        self.planner.t_his.save_csv_data()
        self.supervisor.save_intervention_list()


class RandomPlanner:
    def __init__(self, conf, name="RandoPlanner"):
        self.d_max = conf.max_steer # radians  
        self.name = name
        self.speed = conf.vehicle_speed

    def plan(self, pos):
        steering = np.random.uniform(-self.d_max, self.d_max)
        return np.array([steering, self.speed])



def main():
    # sim = PreRosSim()
    # sim.run_test()

    # sim = PreRosTraining()
    # sim.run_training()

    sim = PreRosTesting()
    sim.run_testing()



if __name__ == '__main__':
    main()






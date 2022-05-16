import time
import numpy as np
# Test Functions

def run_continuous_wrapper_test(env, vehicle, testing_params, show=False):
    crashes = 0
    completes = 0
    lap_times = [] 
    start = time.time()

    observation = env.reset()
    for i in range(testing_params.n_laps):
        action = vehicle.plan

        while not done:
            action = vehicle.plan(obs)
            sim_steps = conf.sim_steps
            while sim_steps > 0 and not done:
                obs, step_reward, done, _ = env.step(action[None, :])
                sim_steps -= 1

            if show:
                # env.render(mode='human')
                env.render(mode='human_fast')
 
        # env.sim.agents[0].history.plot_history()
        r = find_conclusion(obs, start)

        if r == 1:
            completes += 1
            lap_times.append(env.lap_times[0])
        else:
            crashes += 1

        # env.save_traj(f"Traj_{i}_{vehicle.name}")

    success_rate = (completes / (completes + crashes) * 100)
    if len(lap_times) > 0:
        avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
    else:
        avg_times, std_dev = 0, 0

    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {success_rate:.2f} %")
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

    eval_dict = {}
    eval_dict['success_rate'] = float(success_rate)
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)

    print(f"Finished running test and saving file with results.")

    return eval_dict


def evaluate_vehicle(env, vehicle, conf, show=False):
    crashes = 0
    completes = 0
    lap_times = [] 
    start = time.time()
    reset_pose = np.zeros(3)[None, :]

    for i in range(conf.test_n):
        obs, step_reward, done, info = env.reset(reset_pose)
        while not done:
            action = vehicle.plan(obs)
            sim_steps = conf.sim_steps
            while sim_steps > 0 and not done:
                obs, step_reward, done, _ = env.step(action[None, :])
                sim_steps -= 1

            if show:
                # env.render(mode='human')
                env.render(mode='human_fast')
 
        # env.sim.agents[0].history.plot_history()
        r = find_conclusion(obs, start)

        if r == 1:
            completes += 1
            lap_times.append(env.lap_times[0])
        else:
            crashes += 1

        # env.save_traj(f"Traj_{i}_{vehicle.name}")

    success_rate = (completes / (completes + crashes) * 100)
    if len(lap_times) > 0:
        avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
    else:
        avg_times, std_dev = 0, 0

    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {success_rate:.2f} %")
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

    eval_dict = {}
    eval_dict['success_rate'] = float(success_rate)
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)

    print(f"Finished running test and saving file with results.")

    return eval_dict

def evaluate_kernel_vehicle(env, vehicle, conf, show=False):
    crashes = 0
    completes = 0
    lap_times = [] 
    start = time.time()
    interventions = []

    for i in range(conf.test_n):
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        while not done:
            action = vehicle.plan(obs)
            sim_steps = conf.sim_steps
            while sim_steps > 0 and not done:
                obs, step_reward, done, _ = env.step(action[None, :])
                sim_steps -= 1

            if show:
                env.render(mode='human_fast')

        # env.sim.agents[0].history.plot_history()
        r = find_conclusion(obs, start)
        interventions.append(vehicle.interventions)
        print(f"Interventions: {vehicle.interventions}")
        vehicle.interventions = 0

        if r == 1:
            completes += 1
            lap_times.append(env.lap_times[0])
        else:
            crashes += 1

        # vehicle.plot_safe_history(f"Traj_{i}_{vehicle.name}")


    success_rate = (completes / (completes + crashes) * 100)
    if len(lap_times) > 0:
        avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
        avg_interventions = np.mean(interventions)
    else:
        avg_times, std_dev = 0, 0


    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {success_rate:.2f} %")
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")
    print(f"Interventions Avg: {avg_interventions}")

    eval_dict = {}
    eval_dict['success_rate'] = float(success_rate)
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)
    eval_dict['avg_interventions'] = float(avg_interventions)
    eval_dict['std_inters'] = float(np.std(interventions))

    print(f"Finished running test and saving file with results.")

    return eval_dict


def render_kernel_eval(env, vehicle, conf, show=False):
    crashes = 0
    completes = 0
    lap_times = [] 
    start = time.time()
    interventions = []

    for i in range(conf.test_n):
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        while not done:
            action = vehicle.plan(obs)
            sim_steps = conf.sim_steps
            while sim_steps > 0 and not done:
                obs, step_reward, done, _ = env.step(action[None, :])
                sim_steps -= 1

            if show:
                env.render(mode='human_fast')

        # env.sim.agents[0].history.plot_history()
        r = find_conclusion(obs, start)
        interventions.append(vehicle.interventions)
        print(f"Interventions: {vehicle.interventions}")
        vehicle.interventions = 0

        if r == 1:
            completes += 1
            lap_times.append(env.lap_times[0])
        else:
            crashes += 1

        # vehicle.plot_safe_history(f"Traj_{i}_{vehicle.name}")

        env.render_trajectory(vehicle.planner.path, f"Traj_{i}", vehicle.safe_history)
        vehicle.safe_history.save_safe_history(vehicle.planner.path, f"Traj_{i}")

    success_rate = (completes / (completes + crashes) * 100)
    if len(lap_times) > 0:
        avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
        avg_interventions = np.mean(interventions)
    else:
        avg_times, std_dev = 0, 0


    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {success_rate:.2f} %")
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")
    print(f"Interventions Avg: {avg_interventions}")

    eval_dict = {}
    eval_dict['success_rate'] = float(success_rate)
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)
    eval_dict['avg_interventions'] = float(avg_interventions)
    eval_dict['std_inters'] = float(np.std(interventions))

    print(f"Finished running test and saving file with results.")

    return eval_dict


def train_baseline_vehicle(env, vehicle, conf, show=False):
    start_time = time.time()
    state, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    print(f"Starting Baseline Training: {vehicle.name}")
    crash_counter = 0

    ep_steps = 0 
    lap_counter = 0
    for n in range(conf.baseline_train_n + conf.buffer_n):
        state['reward'] = set_reward(state)
        action = vehicle.plan(state)
        sim_steps = conf.sim_steps
        while sim_steps > 0 and not done:
            s_prime, r, done, _ = env.step(action[None, :])
            sim_steps -= 1

        state = s_prime
        if n > conf.buffer_n:
            vehicle.agent.train(2)
        if show:
            env.render('human_fast')
        
        if done or ep_steps > conf.max_steps:
            s_prime['reward'] = set_reward(s_prime) 
            vehicle.done_entry(s_prime)

            print(f"{n}::Lap done {lap_counter} -> FinalR: {s_prime['reward']} -> LapTime {env.lap_times[0]:.2f} -> TotalReward: {vehicle.t_his.rewards[vehicle.t_his.ptr-1]:.2f}")
            lap_counter += 1
            ep_steps = 0 
            if state['reward'] == -1:
                crash_counter += 1

            state, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
            
        ep_steps += 1

    vehicle.t_his.print_update(True)
    vehicle.t_his.save_csv_data()
    vehicle.agent.save(vehicle.path)

    train_time = time.time() - start_time
    print(f"Finished Training: {vehicle.name} in {train_time} seconds")
    print(f"Crashes: {crash_counter}")

    return train_time, crash_counter


def train_kernel_vehicle(env, vehicle, conf, show=False):
    start_time = time.time()
    state, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    print(f"Starting KernelSSS Training: {vehicle.name}")
    crash_counter = 0

    ep_steps = 0 
    lap_counter = 0
    state['reward'] = 0 # unless changed
    for n in range(conf.kernel_train_n+ conf.buffer_n):
        action = vehicle.plan(state)
        sim_steps = conf.sim_steps
        while sim_steps > 0 and not done:
            s_prime, r, done, _ = env.step(action[None, :])
            sim_steps -= 1

        state = s_prime
        state['reward'] = 0 # unless changed
        if n > conf.buffer_n:
            vehicle.planner.agent.train(2)

        if show:
            # env.render('human_fast')
            env.render('human')
        
        if s_prime['collisions'][0] == 1:
            print(f"COLLISION:: Lap done {lap_counter} -> {env.lap_times[0]} -> Inters: {vehicle.ep_interventions}")
            state, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
            lap_counter += 1
            s_prime['reward'] = set_reward(s_prime) # -1 in this position
            vehicle.done_entry(s_prime, env.lap_times[0])
            ep_steps = 0 

        if done or ep_steps > conf.max_steps:
            s_prime['reward'] = set_reward(s_prime) # always lap finished=1 at this position
            vehicle.lap_complete(env.lap_times[0])
            print(f"{n}::Lap done {lap_counter} -> {env.lap_times[0]} -> Inters: {vehicle.intervention_list[-1]} -> FinalR: {s_prime['reward']} -> TotalReward: {vehicle.planner.t_his.rewards[vehicle.planner.t_his.ptr-1]:.2f}")
            lap_counter += 1
            # if show:
            #     env.render(wait=False)

            env.data_reset()
            done = False
            ep_steps = 0 
            
        ep_steps += 1

    vehicle.planner.t_his.print_update(True)
    vehicle.planner.t_his.save_csv_data()
    vehicle.planner.agent.save(vehicle.planner.path)
    vehicle.save_intervention_list()

    train_time = time.time() - start_time
    print(f"Finished Training: {vehicle.planner.name} in {train_time} seconds")
    print(f"Crashes: {crash_counter}")

    return train_time, crash_counter



def find_conclusion(s_p, start):
    laptime = s_p['lap_times'][0]
    if s_p['lap_counts'][0] == 1:
        print(f'Complete --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
        return 1
    else:
        print(f'Collision --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
        return 0



# def find_conclusion(s_p, start):
#     laptime = s_p['lap_times'][0]
#     if s_p['collisions'][0] == 1:
#         print(f'Collision --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
#         return -1
#     elif s_p['lap_counts'][0] == 1:
#         print(f'Complete --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
#         return 1
#     else:
#         print("No conclusion: Awkward palm trees")
#         # print(s_p)
#     return 0


def set_reward(s_p):
    if s_p['collisions'][0] == 1:
        return -1
    elif s_p['lap_counts'][0] == 1:
        return 1
    return 0


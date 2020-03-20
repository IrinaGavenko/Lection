import gym
import numpy
import json
from snn2 import *

from numpy import sin, cos, arcsin, arccos
from random import Random
from math import *

from neat import PopulationPool


def encode(x): # 13 spikeTrains to input
    return [
        [] if x[0] < 0 else [10 - int(x[0] * 10)],
        [] if x[0] > 0 else [10 + int(x[0] * 10)],
        [] if x[1] < 0 else [10 - int(x[1] * 10)],
        [] if x[1] > 0 else [10 + int(x[1] * 10)],
        [] if x[2] < 0 else [10 - int(x[2] * 10)],
        [] if x[2] > 0 else [10 + int(x[2] * 10)],
        [] if x[3] < 0 else [10 - int(x[3] * 10)],
        [] if x[3] > 0 else [10 + int(x[3] * 10)],
        [] if x[4] < 0 else [10 - int(x[3] * 10)],
        [] if x[4] > 0 else [10 + int(x[4] * 10)],
        [] if x[5] < 0 else [10 - int(x[5] * 10)],
        [] if x[5] > 0 else [10 + int(x[5] * 10)],
        [0]
    ]


def evaluate_net(env, net):
    observation = env.reset() # initial
    reward = None
    for t in range(1000): # propagate
        env.render(mode='rgb_array')
        res = net.propagateFull(encode(observation))
        max_i = -1
        max_r = None
        for i, r in enumerate(res):
            if max_i == -1 and len(r) > 0:
                max_i = i
                max_r = r
            elif max_i != -1 and len(r) > 0 and r[0] < max_r[0]:
                max_i = i
                max_r = r

        action = max_i
        observation, temp_reward, done, info = env.step(action)
        if reward is None:
            reward = temp_reward
        else:
            reward = max([reward, temp_reward])
        # reward = -(observation[0]) - cos(arccos(observation[0]) + arccos(observation[2]))

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    return reward

def env_generator():
    env = gym.make('Acrobot-v1')
    env.metadata['video.frames_per_second'] = 1
    return env

if __name__ == '__main__':
    net_config = {
        'inputs_num': 13,
        'hidden_num': 6,
        'outputs_num': 3, # == env.action_space
        'synapsesPerConnection': 3,
        'speciation_threshold': 3.0,
        'percentage_to_save': 0.8,
        'fitness_threshold': 100000.0, # to stop evolution
        'crossover_reenable_connection_rate': 0.25, 
        'connection_mutation_rate': 0.8,
        'connection_perturbation_rate': 0.9,
        'add_node_mutation_rate': 0.03,
        'add_connection_mutation_rate': 0.5,
    }
    population_size = 3
    generations_number = 2

    population_pool = PopulationPool(population_size, net_config, env_generator, evaluate_net)
    population_pool.run(generations_number) # 100 generations

    dict_nets = [ net.serialize() for net in population_pool.population]
    with open(file='dump.json', mode='w', encoding='utf-8') as f:
        json.dump(dict_nets, f, indent=2)

    print(population_pool.best_net.serialize())
    evaluate_net(env_generator(), population_pool.best_net) # for the best_net
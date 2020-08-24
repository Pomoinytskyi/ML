import gym
import time
import argparse

import numpy  as np
import torch

from lib import wrappers
from lib import dqn_model

import collections
from datetime import datetime
import random

Default_Env_Name = "PongNoFrameskip-v4"
Fps = 25

if __name__ == "__main__":
    rnd = random.Random()
    rnd.seed(datetime.now())

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required = True, help = "Model file to load")
    parser.add_argument("-e", "--env", default = Default_Env_Name, help = "Envirinment name. Defauit = "+Default_Env_Name)
    parser.add_argument("-r", "--record", help = "Directory for fideo")
    parser.add_argument("--no-vis", default = True, dest='vis', help = "Disable visualization", action='store_false')
    args = parser.parse_args()

    print("Arguments:")
    print(args)

    
    env = wrappers.make_env(args.env)            
    env.seed(rnd.randint(1, 100000))
    env.action_space.seed(rnd.randint(1, 100000))

    if args.record:
        gym.wrappers.Monitor(env, args.record, force=True)
    
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location = lambda stg,_ : stg)
    net.load_state_dict(state)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.vis:
            env.render()

        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1

        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

        if args.vis:
            delta = 1/Fps - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    
    print("Total reward: %2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()

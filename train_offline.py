import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
from pathlib import Path
import yaml
# offline data: from D4RL benchmark
import d4rl

from algo.PAR_offline import PAR
import algo.utils as utils
from envs.env_utils import call_terminal_func
from envs.common import call_env
from tensorboardX import SummaryWriter


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))
    
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="PAR", help='policy to use, support PAR')
    parser.add_argument("--env", default="halfcheetah")
    parser.add_argument('--type', default="medium", help='dataset type used in source domain')
    parser.add_argument("--seed", default=0, type=int)            
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--distance", default="Euclidean")          # support different distance measurement
    parser.add_argument('--beta', help='penalty coefficient', default=1.0, type=float)
    parser.add_argument("--weight", default=2.5, type=float)         # weighting coefficient for behavior cloning
    parser.add_argument("--layernorm", action="store_true")
    
    args = parser.parse_args()

    with open(f"{str(Path(__file__).parent.absolute())}/config/{args.env}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("------------------------------------------------------------")
    print("------------------Start Offline Training!-------------------")
    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    outdir = args.dir + '/' + args.env + '/r' + str(args.seed)
    writer = SummaryWriter('{}/tb'.format(outdir))
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    if 'morph' in args.env:
        src_env_name = args.env.split('_')[0]
    else:
        src_env_name = args.env
    
    # train env
    if 'walker' in args.env:
        src_env_name += '2d-' + args.type + '-v2'
    else:
        src_env_name += '-' + args.type + '-v2'
    src_env = gym.make(src_env_name)
    src_env.seed(args.seed)
    # test env
    tar_env = call_env(config['tar_env_config'])
    tar_env.seed(args.seed)
    # eval env
    tar_eval_env = call_env(config['tar_env_config'])
    tar_eval_env.seed(args.seed + 100)

    # seed all
    tar_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    state_dim = src_env.observation_space.shape[0]
    action_dim = src_env.action_space.shape[0] 
    max_action = float(src_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config['beta'] = args.beta
    config['distance'] = args.distance
    config['weight'] = args.weight
    config['layernorm'] = args.layernorm

    config.update({
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
    })

    policy = PAR(config, device)
    
    ## write logs to record training parameters
    with open(outdir + 'log.txt','w') as f:
        f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))

    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    # load offline dataset
    src_replay_buffer.convert_D4RL(d4rl.qlearning_dataset(src_env))
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    eval_cnt = 0
    
    eval_src_return = eval_policy(policy, src_env, eval_cnt=eval_cnt)
    eval_tar_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
    eval_cnt += 1

    tar_state, tar_done = tar_env.reset(), False
    tar_episode_timesteps, tar_episode_num, tar_episode_reward = 0, 0, 0

    for t in range(int(config['max_step'])):
        
        # interaction with tar env
        if t % config['tar_env_interact_freq'] == 0:
            tar_episode_timesteps += 1
            tar_action = policy.select_action(np.array(tar_state), test=False)

            tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)
            tar_done_bool = float(tar_done) if tar_episode_timesteps < src_env._max_episode_steps else 0

            tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

            tar_state = tar_next_state
            tar_episode_reward += tar_reward

        policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
        
        if tar_done:
            print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, tar_episode_num+1, tar_episode_timesteps, tar_episode_reward))
            writer.add_scalar('train/target return', tar_episode_reward, global_step = t+1)

            tar_state, tar_done = tar_env.reset(), False
            tar_episode_reward = 0
            tar_episode_timesteps = 0
            tar_episode_num += 1

        if (t + 1) % config['eval_freq'] == 0:
            src_eval_return = eval_policy(policy, src_env, eval_cnt=eval_cnt)
            tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
            writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
            writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
            eval_cnt += 1

            if args.save_model:
                policy.save('{}/models/model'.format(outdir))
    writer.close()

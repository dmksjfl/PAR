import numpy as np
import gym

class EnvSampler:
    def __init__(self, env:gym.Env, max_path_length: int = 1000):
        self.env = env

        self.path_length        = 0
        self.current_state      = None
        self.max_path_length    = max_path_length
        self.path_rewards       = []
        self.sum_reward         = 0

    def sample(self, agent = None, with_noise: bool = True, render: bool = False, optimistic: bool = False):
        if self.current_state is None:
            self.current_state = self.env.reset()
        cur_state = self.current_state

        if render:
            self.env.render()

        if agent is None:
            action = self.env.action_space.sample()
        else:
            if optimistic:
                try:
                    action = agent.sample_optimistic_action(cur_state, with_noise)
                except:
                    raise NotImplementedError('This agent has no optimistic sampling function')
            else:
                action = agent.sample_action(cur_state, with_noise)
        
        next_state, reward, terminal, info = self.env.step(action)

        self.path_length += 1
        self.sum_reward += reward

        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_state, action, reward, terminal, next_state, info

    def evaluate(self, agent, num_episode, return_full: bool = False, render: bool = False) -> float:
        eval_scores = []
        for _ in range(num_episode):
            epi_r = 0
            s = self.env.reset()
            done = False
            while not done:
                if render:
                    self.env.render()
                a = agent.sample_action(s, False)
                s, r, done, info = self.env.step(a)
                epi_r += r
            eval_scores.append(epi_r)

        # reset the episode
        self.current_state = None
        self.path_length = 0
        self.sum_reward = 0
        
        if return_full:
            return eval_scores
        else:
            return np.mean(eval_scores)
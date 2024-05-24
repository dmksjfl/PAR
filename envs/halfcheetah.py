from typing import List, Dict, Union, Tuple
import random
import numpy as np
import gym



class HC_Dynamics_Wrapper(gym.Wrapper):
    def __init__(self, param_dict: Dict = dict()):
        super().__init__(gym.make('HalfCheetah-v2'))
        self.name               = 'HalfCheetah'
        
        self.params             = list(param_dict.keys())
        self.param_dict         = param_dict
        self.initial_param_dict = {param: [] for param in self.params}

        self.current_param_scale = dict()

        for param in self.params:
            if param == 'foot mass':
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[4])
            elif param == 'shin mass':
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[3])
            elif param == 'torso mass':
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[5])
            elif param == 'foot fric':
                self.initial_param_dict[param].append(self.unwrapped.model.geom_friction[5][0])
            elif param == 'damping':
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[3])
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[4])
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[5])
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[6])
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[7])
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[8])
            elif param == 'mass':
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[1])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[2])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[3])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[4])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[5])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[6])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[7])
            elif param == 'bt jnt lower limit':
                self.initial_param_dict[param].append(self.unwrapped.model.jnt_range[3][0])
            elif param == 'bt jnt upper limit':
                self.initial_param_dict[param].append(self.unwrapped.model.jnt_range[3][1])
            elif param == 'ff jnt lower limit':
                self.initial_param_dict[param].append(self.unwrapped.model.jnt_range[8][0])
            elif param == 'ff jnt upper limit':
                self.initial_param_dict[param].append(self.unwrapped.model.jnt_range[8][1])
            else:
                raise NotImplementedError(f"{param} is not adjustable in HalfCheetah")

            self.current_param_scale[param]    = 1

    def set_params(self, param_scales: Dict) -> None:
        assert len(param_scales) == len(self.params), 'Length of new params must align the initilization params'
        for param, scale in list(param_scales.items()):
            if param == 'foot mass':
                self.unwrapped.model.body_mass[4] = self.initial_param_dict[param][-1] * scale
                self.unwrapped.model.body_mass[7] = self.initial_param_dict[param][-1] * scale
            elif param == 'foot fric':
                self.unwrapped.model.geom_friction[5][0] = self.initial_param_dict[param][-1] * scale
                self.unwrapped.model.geom_friction[8][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'torso mass':
                self.unwrapped.model.body_mass[1] = self.initial_param_dict[param][-1] * scale
            elif param == 'shin mass':
                self.unwrapped.model.body_mass[3] = self.initial_param_dict[param][-1] * scale
                self.unwrapped.model.body_mass[6] = self.initial_param_dict[param][-1] * scale
            elif param == 'damping':
                self.unwrapped.model.dof_damping[3]  = self.initial_param_dict[param][0] * scale
                self.unwrapped.model.dof_damping[4]  = self.initial_param_dict[param][1] * scale
                self.unwrapped.model.dof_damping[5]  = self.initial_param_dict[param][2] * scale
                self.unwrapped.model.dof_damping[6]  = self.initial_param_dict[param][3] * scale
                self.unwrapped.model.dof_damping[7]  = self.initial_param_dict[param][4] * scale
                self.unwrapped.model.dof_damping[8]  = self.initial_param_dict[param][5] * scale
            elif param == 'mass':
                self.unwrapped.model.body_mass[1] =  self.initial_param_dict[param][0] * scale
                self.unwrapped.model.body_mass[2] =  self.initial_param_dict[param][1] * scale
                self.unwrapped.model.body_mass[3] =  self.initial_param_dict[param][2] * scale
                self.unwrapped.model.body_mass[4] =  self.initial_param_dict[param][3] * scale
                self.unwrapped.model.body_mass[5] =  self.initial_param_dict[param][4] * scale
                self.unwrapped.model.body_mass[6] =  self.initial_param_dict[param][5] * scale
                self.unwrapped.model.body_mass[7] =  self.initial_param_dict[param][6] * scale
            elif param == 'ff jnt lower limit':
                self.unwrapped.model.jnt_range[8][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'ff jnt upper limit':
                self.unwrapped.model.jnt_range[8][1] = self.initial_param_dict[param][-1] * scale
            elif param == 'bt jnt lower limit':
                self.unwrapped.model.jnt_range[3][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'bt jnt upper limit':
                self.unwrapped.model.jnt_range[3][1] = self.initial_param_dict[param][-1] * scale
    
            self.current_param_scale[param] = scale

    def resample_params(self) -> None:
        new_scales = {}
        for param, bound_or_possible_values in list(self.param_dict.items()):
            if len(bound_or_possible_values) == 2:
                new_scales[param] = random.uniform(
                    bound_or_possible_values[0],
                    bound_or_possible_values[1]
                )
            else:
                new_scales[param] = random.choice(bound_or_possible_values)
        self.set_params(new_scales)

    def reset(self, resample: bool = True) -> np.array:
        if resample:
            self.resample_params()
        return self.env.reset()

    @property
    def current_param_scales(self) -> Dict:
        return self.current_param_scale

    @property
    def current_flat_scale(self) -> List:
        return list(self.current_param_scale.values())

    @property
    def action_bound(self) -> float:
        return self.env.action_space.high[0]



class HC_Speed_and_Dynamics_Wrapper(gym.Wrapper):
    def __init__(self, param_dict: Dict = dict()):
        super().__init__(gym.make('HalfCheetah-v2'))
        self.name               = 'HalfCheetah'
        
        self.params             = list(param_dict.keys())
        self.param_dict         = param_dict
        self.initial_param_dict = {param: [] for param in self.params}

        self.current_param_scale = dict()

        for param in self.params:
            if param == 'speed':
                self.initial_param_dict[param].append(param_dict[param])
                self.speed_requirement = param_dict[param]
            elif param == 'foot mass':
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[4])
            elif param == 'shin mass':
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[3])
            elif param == 'torso mass':
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[5])
            elif param == 'foot fric':
                self.initial_param_dict[param].append(self.unwrapped.model.geom_friction[5][0])
            elif param == 'damping':
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[3])
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[4])
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[5])
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[6])
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[7])
                self.initial_param_dict[param].append(self.unwrapped.model.dof_damping[8])
            elif param == 'mass':
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[1])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[2])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[3])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[4])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[5])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[6])
                self.initial_param_dict[param].append(self.unwrapped.model.body_mass[7])
            elif param == 'bt jnt lower limit':
                self.initial_param_dict[param].append(self.unwrapped.model.jnt_range[3][0])
            elif param == 'bt jnt upper limit':
                self.initial_param_dict[param].append(self.unwrapped.model.jnt_range[3][1])
            elif param == 'ff jnt lower limit':
                self.initial_param_dict[param].append(self.unwrapped.model.jnt_range[8][0])
            elif param == 'ff jnt upper limit':
                self.initial_param_dict[param].append(self.unwrapped.model.jnt_range[8][1])
            else:
                raise NotImplementedError(f"{param} is not adjustable in HalfCheetah")

            self.current_param_scale[param]    = 1

    def step(self, action):
        observation, reward, terminated, info = self.env.step(action)
        if info['reward_run'] < self.speed_requirement:
            reward = info['reward_ctrl']
        return observation, reward, terminated, info

    def set_params(self, param_scales: Dict) -> None:
        assert len(param_scales) == len(self.params), 'Length of new params must align the initilization params'
        for param, scale in list(param_scales.items()):
            if param == 'foot mass':
                self.unwrapped.model.body_mass[4] = self.initial_param_dict[param][-1] * scale
                self.unwrapped.model.body_mass[7] = self.initial_param_dict[param][-1] * scale
            elif param == 'foot fric':
                self.unwrapped.model.geom_friction[5][0] = self.initial_param_dict[param][-1] * scale
                self.unwrapped.model.geom_friction[8][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'torso mass':
                self.unwrapped.model.body_mass[1] = self.initial_param_dict[param][-1] * scale
            elif param == 'shin mass':
                self.unwrapped.model.body_mass[3] = self.initial_param_dict[param][-1] * scale
                self.unwrapped.model.body_mass[6] = self.initial_param_dict[param][-1] * scale
            elif param == 'damping':
                self.unwrapped.model.dof_damping[3]  = self.initial_param_dict[param][0] * scale
                self.unwrapped.model.dof_damping[4]  = self.initial_param_dict[param][1] * scale
                self.unwrapped.model.dof_damping[5]  = self.initial_param_dict[param][2] * scale
                self.unwrapped.model.dof_damping[6]  = self.initial_param_dict[param][3] * scale
                self.unwrapped.model.dof_damping[7]  = self.initial_param_dict[param][4] * scale
                self.unwrapped.model.dof_damping[8]  = self.initial_param_dict[param][5] * scale
            elif param == 'mass':
                self.unwrapped.model.body_mass[1] =  self.initial_param_dict[param][0] * scale
                self.unwrapped.model.body_mass[2] =  self.initial_param_dict[param][1] * scale
                self.unwrapped.model.body_mass[3] =  self.initial_param_dict[param][2] * scale
                self.unwrapped.model.body_mass[4] =  self.initial_param_dict[param][3] * scale
                self.unwrapped.model.body_mass[5] =  self.initial_param_dict[param][4] * scale
                self.unwrapped.model.body_mass[6] =  self.initial_param_dict[param][5] * scale
                self.unwrapped.model.body_mass[7] =  self.initial_param_dict[param][6] * scale
            elif param == 'ff jnt lower limit':
                self.unwrapped.model.jnt_range[8][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'ff jnt upper limit':
                self.unwrapped.model.jnt_range[8][1] = self.initial_param_dict[param][-1] * scale
            elif param == 'bt jnt lower limit':
                self.unwrapped.model.jnt_range[3][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'bt jnt upper limit':
                self.unwrapped.model.jnt_range[3][1] = self.initial_param_dict[param][-1] * scale
    
            self.current_param_scale[param] = scale

    def resample_params(self) -> None:
        new_scales = {}
        for param, bound_or_possible_values in list(self.param_dict.items()):
            if len(bound_or_possible_values) == 2:
                new_scales[param] = random.uniform(
                    bound_or_possible_values[0],
                    bound_or_possible_values[1]
                )
            else:
                new_scales[param] = random.choice(bound_or_possible_values)
        self.set_params(new_scales)

    def reset(self, resample: bool = True) -> np.array:
        if resample:
            self.resample_params()
        return self.env.reset()

    @property
    def current_param_scales(self) -> Dict:
        return self.current_param_scale

    @property
    def current_flat_scale(self) -> List:
        return list(self.current_param_scale.values())

    @property
    def action_bound(self) -> float:
        return self.env.action_space.high[0]


if __name__ == '__main__':
    env = HC_Speed_and_Dynamics_Wrapper({})
    s = env.reset()
    while True:
        a = env.action_space.sample()
        s = env.step(a)

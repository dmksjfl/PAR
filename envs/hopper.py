from typing import List, Dict, Union, Tuple
import random
import numpy as np
import gym



class HP_Dynamics_Wrapper(gym.Wrapper):
    def __init__(self, param_dict: Dict = dict()):
        super().__init__(gym.make('Hopper-v2'))
        self.name               = 'Hopper'

        self.params             = list(param_dict.keys())
        self.param_dict         = param_dict
        self.initial_param_dict = {param: [] for param in self.params}

        self.current_param_scale = dict()

        for param in self.params:
            if param == 'torso mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[1])
            elif param == 'thigh mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[2])
            elif param == 'leg mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[3])
            elif param == 'foot mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[4])
            elif param == 'foot fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[-1][0])
            elif param == 'damping':
                self.initial_param_dict[param].append(self.env.model.dof_damping[-3])
                self.initial_param_dict[param].append(self.env.model.dof_damping[-2])
                self.initial_param_dict[param].append(self.env.model.dof_damping[-1])
            elif param == 'mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[1])
                self.initial_param_dict[param].append(self.env.model.body_mass[2])
                self.initial_param_dict[param].append(self.env.model.body_mass[3])
                self.initial_param_dict[param].append(self.env.model.body_mass[4])
            elif param == 'foot jnt lower limit':
                self.initial_param_dict[param].append(self.env.model.jnt_range[5][0])
            elif param == 'foot jnt upper limit':
                self.initial_param_dict[param].append(self.env.model.jnt_range[5][1])
            elif param == 'leg jnt lower limit':
                self.initial_param_dict[param].append(self.env.model.jnt_range[4][0])
            elif param == 'torso jnt lower limit':
                self.initial_param_dict[param].append(self.env.model.jnt_range[3][0])
            else:
                raise NotImplementedError(f"{param} is not adjustable in Hopper")

            self.current_param_scale[param]    = 1

    def set_params(self, param_scales: Dict) -> None:
        assert len(param_scales) == len(self.params), 'Length of new params must align the initilization params'
        for param, scale in list(param_scales.items()):
            if param == 'torso mass':
                self.env.model.body_mass[1] = self.initial_param_dict[param][-1] * scale
            elif param == 'thigh mass':
                self.env.model.body_mass[2] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg mass':
                self.env.model.body_mass[3] = self.initial_param_dict[param][-1] * scale
            elif param == 'foot mass':
                self.env.model.body_mass[4] = self.initial_param_dict[param][-1] * scale
            elif param == 'foot fric':
                self.env.model.geom_friction[-1][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'damping':
                self.env.model.dof_damping[-3] = self.initial_param_dict[param][0] * scale
                self.env.model.dof_damping[-2] = self.initial_param_dict[param][1] * scale
                self.env.model.dof_damping[-1] = self.initial_param_dict[param][2] * scale
            elif param == 'mass':
                self.env.model.body_mass[1] = self.initial_param_dict[param][0] * scale
                self.env.model.body_mass[2] = self.initial_param_dict[param][1] * scale
                self.env.model.body_mass[3] = self.initial_param_dict[param][2] * scale
                self.env.model.body_mass[4] = self.initial_param_dict[param][3] * scale
            elif param == 'foot jnt lower limit':
                self.env.model.jnt_range[5][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'foot jnt upper limit':
                self.env.model.jnt_range[5][1] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg jnt lower limit':
                self.env.model.jnt_range[4][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'torso jnt lower limit':
                self.env.model.jnt_range[3][0] = self.initial_param_dict[param][-1] * scale

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

    def reset(self, resample: bool=True) -> np.array:
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




class HP_Nonstationary_Wrapper(gym.Wrapper):
    def __init__(self, param_dict: Dict = dict(), change_freq: int = 50):
        super().__init__(gym.make('Hopper-v2'))
        self.name               = 'Hopper'

        self.params             = list(param_dict.keys())
        self.param_dict         = param_dict
        self.initial_param_dict = {param: [] for param in self.params}

        self.current_param_scale = dict()

        for param in self.params:
            if param == 'torso mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[1])
            elif param == 'thigh mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[2])
            elif param == 'leg mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[3])
            elif param == 'foot mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[4])
            elif param == 'foot fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[-1][0])
            elif param == 'damping':
                self.initial_param_dict[param].append(self.env.model.dof_damping[-3])
                self.initial_param_dict[param].append(self.env.model.dof_damping[-2])
                self.initial_param_dict[param].append(self.env.model.dof_damping[-1])
            elif param == 'mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[1])
                self.initial_param_dict[param].append(self.env.model.body_mass[2])
                self.initial_param_dict[param].append(self.env.model.body_mass[3])
                self.initial_param_dict[param].append(self.env.model.body_mass[4])
            else:
                raise NotImplementedError(f"{param} is not adjustable in Hopper")

            self.current_param_scale[param]    = 1

        self.episode_step   =   0
        self.change_freq    =   change_freq

    def set_params(self, param_scales: Dict) -> None:
        assert len(param_scales) == len(self.params), 'Length of new params must align the initilization params'
        for param, scale in list(param_scales.items()):
            if param == 'torso mass':
                self.env.model.body_mass[1] = self.initial_param_dict[param][-1] * scale
            elif param == 'thigh mass':
                self.env.model.body_mass[2] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg mass':
                self.env.model.body_mass[3] = self.initial_param_dict[param][-1] * scale
            elif param == 'foot mass':
                self.env.model.body_mass[4] = self.initial_param_dict[param][-1] * scale
            elif param == 'foot fric':
                self.env.model.geom_friction[-1][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'damping':
                self.env.model.dof_damping[-3] = self.initial_param_dict[param][0] * scale
                self.env.model.dof_damping[-2] = self.initial_param_dict[param][1] * scale
                self.env.model.dof_damping[-1] = self.initial_param_dict[param][2] * scale
            elif param == 'mass':
                self.env.model.body_mass[1] = self.initial_param_dict[param][0] * scale
                self.env.model.body_mass[2] = self.initial_param_dict[param][1] * scale
                self.env.model.body_mass[3] = self.initial_param_dict[param][2] * scale
                self.env.model.body_mass[4] = self.initial_param_dict[param][3] * scale

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

    def reset(self, resample: bool=True) -> np.array:
        if resample:
            self.resample_params()
        self.episode_step = 0
        return self.env.reset()

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        if self.episode_step % self.change_freq == 0:
            self.resample_params()
        self.episode_step += 1
        return super().step(action)

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
    env = HP_Dynamics_Wrapper()

    for _ in range(50):
        done = False
        s    = env.reset()
        while not done:
            env.render()
            # a = np.zeros_like(env.action_space.sample())
            a = env.action_space.sample()
            s, r, done, info = env.step(a)

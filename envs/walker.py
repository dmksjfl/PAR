from typing import List, Dict, Union, Tuple
import random
import numpy as np
import gym



class WK_Dynamics_Wrapper(gym.Wrapper):
    def __init__(self, param_dict: Dict = dict()):
        super().__init__(gym.make('Walker2d-v2'))

        self.name               = 'Walker'

        self.params             = list(param_dict.keys())
        self.param_dict         = param_dict
        self.initial_param_dict = {param: [] for param in self.params}

        self.current_param_scale = dict()

        for param in self.params:
            if param == 'torso mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[1])
            elif param == 'shin mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[2])
                self.initial_param_dict[param].append(self.env.model.body_mass[5])
            elif param == 'leg mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[3])
                self.initial_param_dict[param].append(self.env.model.body_mass[6])
            elif param == 'foot mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[4])
                self.initial_param_dict[param].append(self.env.model.body_mass[7])
            elif param == 'left foot fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[7][0])
            elif param == 'right foot fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[4][0])
                
            elif param == 'right foot jnt lower limit':
                self.initial_param_dict[param].append(self.env.model.jnt_range[5][0])
            elif param == 'right foot jnt upper limit':
                self.initial_param_dict[param].append(self.env.model.jnt_range[5][1])
            elif param == 'left foot jnt lower limit':
                self.initial_param_dict[param].append(self.env.model.jnt_range[8][0])
            elif param == 'left foot jnt upper limit':
                self.initial_param_dict[param].append(self.env.model.jnt_range[8][1])
            else:
                raise NotImplementedError(f"{param} is not adjustable in Walker")

            self.current_param_scale[param]    = 1

    def set_params(self, param_scales: Dict) -> None:
        assert len(param_scales) == len(self.params), 'Length of new params must align the initilization params'
        for param, scale in list(param_scales.items()):
            if param == 'foot mass':
                self.env.model.body_mass[4] = self.initial_param_dict[param][0] * scale
                self.env.model.body_mass[7] = self.initial_param_dict[param][1] * scale
            elif param == 'torso mass':
                self.env.model.body_mass[1] = self.initial_param_dict[param][-1] * scale
            elif param == 'shin mass':
                self.env.model.body_mass[2] = self.initial_param_dict[param][2] * scale
                self.env.model.body_mass[5] = self.initial_param_dict[param][5] * scale
            elif param == 'leg mass':
                self.env.model.body_mass[3] = self.initial_param_dict[param][3] * scale
                self.env.model.body_mass[6] = self.initial_param_dict[param][6] * scale
            elif param == 'left foot fric':
                self.env.model.geom_friction[7][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'right foot fric':
                self.env.model.geom_friction[4][0] = self.initial_param_dict[param][-1] * scale

            elif param == 'right foot jnt lower limit':
                self.env.model.jnt_range[5][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'right foot jnt upper limit':
                self.env.model.jnt_range[5][1] = self.initial_param_dict[param][-1] * scale
            elif param == 'left foot jnt lower limit':
                self.env.model.jnt_range[8][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'left foot jnt upper limit':
                self.env.model.jnt_range[8][1] = self.initial_param_dict[param][-1] * scale
    
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
from typing import List, Dict, Union, Tuple
import random
import numpy as np
import gym



class AT_Dynamics_Wrapper(gym.Wrapper):
    def __init__(self, param_dict: Dict = dict()):
        super().__init__(gym.make('Ant-v3'))

        self.name               = 'Ant'
        
        self.params             = list(param_dict.keys())
        self.param_dict         = param_dict
        self.initial_param_dict = {param: [] for param in self.params}

        self.current_param_scale = dict()

        for param in self.params:
            if param == 'torso mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[1])

            elif param == 'leg 1 mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[3])
            elif param == 'leg 2 mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[6])
            elif param == 'leg 3 mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[9])
            elif param == 'leg 4 mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[12])
            elif param == 'leg mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[3])
                self.initial_param_dict[param].append(self.env.model.body_mass[6])
                self.initial_param_dict[param].append(self.env.model.body_mass[9])
                self.initial_param_dict[param].append(self.env.model.body_mass[12])

            elif param == 'shin 1 mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[2])
            elif param == 'shin 2 mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[5])
            elif param == 'shin 3 mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[8])
            elif param == 'shin 4 mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[11])
            elif param == 'shin mass':
                self.initial_param_dict[param].append(self.env.model.body_mass[2])
                self.initial_param_dict[param].append(self.env.model.body_mass[5])
                self.initial_param_dict[param].append(self.env.model.body_mass[8])
                self.initial_param_dict[param].append(self.env.model.body_mass[11])

            elif param == 'ankle 1 fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[4][0])
            elif param == 'ankle 2 fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[7][0])
            elif param == 'ankle 3 fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[10][0])
            elif param == 'ankle 4 fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[13][0])

            elif param == 'leg 1 fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[3][0])
            elif param == 'leg 2 fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[6][0])
            elif param == 'leg 3 fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[9][0])
            elif param == 'leg 4 fric':
                self.initial_param_dict[param].append(self.env.model.geom_friction[12][0])

            elif param == 'hip damping':
                self.initial_param_dict[param].append(self.env.model.dof_damping[6])
                self.initial_param_dict[param].append(self.env.model.dof_damping[8])
                self.initial_param_dict[param].append(self.env.model.dof_damping[10])
                self.initial_param_dict[param].append(self.env.model.dof_damping[12])
            elif param == 'ankle damping':
                self.initial_param_dict[param].append(self.env.model.dof_damping[7])
                self.initial_param_dict[param].append(self.env.model.dof_damping[9])
                self.initial_param_dict[param].append(self.env.model.dof_damping[11])
                self.initial_param_dict[param].append(self.env.model.dof_damping[13])

            elif param == 'fl hip lower limit':
                self.initial_param_dict[param].append(self.model.jnt_range[1][0])
            elif param == 'fl hip upper limit':
                self.initial_param_dict[param].append(self.model.jnt_range[1][1])
            
            elif param == 'fl ankle lower limit':
                self.initial_param_dict[param].append(self.model.jnt_range[2][0])
            elif param == 'fl ankle upper limit':
                self.initial_param_dict[param].append(self.model.jnt_range[2][1])

            elif param == 'fr hip lower limit':
                self.initial_param_dict[param].append(self.model.jnt_range[3][0])
            elif param == 'fr hip upper limit':
                self.initial_param_dict[param].append(self.model.jnt_range[3][1])
            
            elif param == 'fr ankle lower limit':
                self.initial_param_dict[param].append(self.model.jnt_range[4][0])
            elif param == 'fr ankle upper limit':
                self.initial_param_dict[param].append(self.model.jnt_range[4][1])

            else:
                raise NotImplementedError(f"{param} is not adjustable in Ant")

            self.current_param_scale[param]    = 1

    def set_params(self, param_scales: Dict) -> None:
        assert len(param_scales) == len(self.params), 'Length of new params must align the initilization params'
        for param, scale in list(param_scales.items()):
            if param == 'torso mass':
                self.env.model.body_mass[1] = self.initial_param_dict[param][-1] * scale

            elif param == 'leg 1 mass':
                self.env.model.body_mass[3] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 2 mass':
                self.env.model.body_mass[6] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 3 mass':
                self.env.model.body_mass[9] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 4 mass':
                self.env.model.body_mass[12] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg mass':
                self.env.model.body_mass[3] = self.initial_param_dict[param][0] * scale
                self.env.model.body_mass[6] = self.initial_param_dict[param][1] * scale
                self.env.model.body_mass[9] = self.initial_param_dict[param][2] * scale
                self.env.model.body_mass[12] = self.initial_param_dict[param][3] * scale

            elif param == 'shin 1 mass':
                self.env.model.body_mass[2] = self.initial_param_dict[param][-1] * scale
            elif param == 'shin 2 mass':
                self.env.model.body_mass[5] = self.initial_param_dict[param][-1] * scale
            elif param == 'shin 3 mass':
                self.env.model.body_mass[8] = self.initial_param_dict[param][-1] * scale
            elif param == 'shin 4 mass':
                self.env.model.body_mass[11] = self.initial_param_dict[param][-1] * scale
            elif param == 'shin mass':
                self.env.model.body_mass[2] = self.initial_param_dict[param][0] * scale
                self.env.model.body_mass[5] = self.initial_param_dict[param][1] * scale
                self.env.model.body_mass[8] = self.initial_param_dict[param][2] * scale
                self.env.model.body_mass[11] = self.initial_param_dict[param][3] * scale

            elif param == 'ankle 1 fric':
                self.env.model.geom_friction[4][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'ankle 2 fric':
                self.env.model.geom_friction[7][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'ankle 3 fric':
                self.env.model.geom_friction[10][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'ankle 4 fric':
                self.env.model.geom_friction[13][0] = self.initial_param_dict[param][-1] * scale

            elif param == 'leg 1 fric':
                self.env.model.geom_friction[3][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 2 fric':
                self.env.model.geom_friction[6][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 3 fric':
                self.env.model.geom_friction[9][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 4 fric':
                self.env.model.geom_friction[12][0] = self.initial_param_dict[param][-1] * scale
            
            elif param == 'hip damping':
                self.env.model.dof_damping[6]  = self.initial_param_dict[param][0] * scale
                self.env.model.dof_damping[8]  = self.initial_param_dict[param][1] * scale
                self.env.model.dof_damping[10]  = self.initial_param_dict[param][2] * scale
                self.env.model.dof_damping[12]  = self.initial_param_dict[param][3] * scale
            elif param == 'ankle damping':
                self.env.model.dof_damping[7]  = self.initial_param_dict[param][0] * scale
                self.env.model.dof_damping[9]  = self.initial_param_dict[param][1] * scale
                self.env.model.dof_damping[11]  = self.initial_param_dict[param][2] * scale
                self.env.model.dof_damping[12]  = self.initial_param_dict[param][3] * scale
            
            elif param == 'fl hip lower limit':
                self.model.jnt_range[1][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'fl hip upper limit':
                self.model.jnt_range[1][1] = self.initial_param_dict[param][-1] * scale

            elif param == 'fl ankle lower limit':
                self.model.jnt_range[2][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'fl ankle upper limit':
                self.model.jnt_range[2][1] = self.initial_param_dict[param][-1] * scale

            elif param == 'fr hip lower limit':
                self.model.jnt_range[3][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'fr hip upper limit':
                self.model.jnt_range[3][1] = self.initial_param_dict[param][-1] * scale

            elif param == 'fr ankle lower limit':
                self.model.jnt_range[4][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'fr ankle upper limit':
                self.model.jnt_range[4][1] = self.initial_param_dict[param][-1] * scale
    
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
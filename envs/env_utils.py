from typing import Callable, Dict, List, Union
import numpy as np
import gym
import torch



def is_terminal_region_for_hp(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
    if isinstance(state, np.ndarray):
        height = state[..., 0:1]
        angle = state[..., 1:2]
        not_done = np.isfinite(state).all(axis=-1)[..., np.newaxis] \
                    & np.abs(state[..., 1:] < 100).all(axis=-1)[..., np.newaxis] \
                    & (height > .7) \
                    & (np.abs(angle) < .2)
    elif isinstance(state, torch.Tensor):
        height = state[..., 0:1]
        angle = state[..., 1:2]
        not_done = torch.isfinite(state).all(dim=-1, keepdim=True) \
                    & torch.abs(state[..., 1:] < 100).all(dim=-1, keepdim=True) \
                    & (height > .7) \
                    & (torch.abs(angle) < .2)
    else:
        raise ValueError
    if return_done:
        done = ~not_done
        return done
    else:
        return not_done

def is_terminal_region_for_hc(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
    if isinstance(state, np.ndarray):
        not_done = np.ones((*state.shape[:-1], 1), dtype=np.bool8)
    elif isinstance(state, torch.Tensor):
        not_done = torch.ones((*state.shape[:-1], 1)).bool()
        not_done = not_done.to(state.device)
    else:
        raise ValueError
        
    if return_done:
        done = ~not_done
        return done
    else:
        return not_done


def is_terminal_region_for_at(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
    if isinstance(state, np.ndarray):
        is_finite = np.isfinite(state).all(-1)[..., np.newaxis]
        is_healthy= (0.2 <= state[..., 0:1]) & (state[..., 0:1] <= 1.0)     # height - z
        not_done = is_finite & is_healthy
    elif isinstance(state, torch.Tensor):
        is_finite = torch.isfinite(state).all(-1, keepdim=True)
        is_healthy= (0.2 <= state[..., 0:1]) & (state[..., 0:1] <= 1.0)
        not_done = is_finite & is_healthy
    else:
        raise ValueError
        
    if return_done:
        done = ~not_done
        return done
    else:
        return not_done


def is_terminal_region_for_sw(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
    if isinstance(state, np.ndarray):
        not_done = np.ones((*state.shape[:-1], 1), dtype=np.bool8)
    elif isinstance(state, torch.Tensor):
        not_done = torch.ones((*state.shape[:-1], 1)).bool()
        not_done = not_done.to(state.device)
    else:
        raise ValueError
        
    if return_done:
        done = ~not_done
        return done
    else:
        return not_done


def is_terminal_region_for_wk(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
    if isinstance(state, np.ndarray):
        height = state[..., 0:1]
        angle = state[..., 1:2]
        not_done = (0.8 < height) & (height < 2.0) & (-1.0 < angle) & (angle < 1.0)
    elif isinstance(state, torch.Tensor):
        height = state[..., 0:1]
        angle = state[..., 1:2]
        not_done = (0.8 < height) & (height < 2.0) & (-1.0 < angle) & (angle < 1.0)
    else:
        raise ValueError
        
    if return_done:
        done = ~not_done
        return done
    else:
        return not_done

def is_terminal_region_for_hm(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
    if isinstance(state, np.ndarray):
        height = state[..., 0:1]
        not_done = (1.0 <= height) & (height <= 2.0)
    elif isinstance(state, torch.Tensor):
        height = state[..., 0:1]
        not_done = (1.0 <= height) & (height <= 2.0)
    else:
        raise ValueError
        
    if return_done:
        done = ~not_done
        return done
    else:
        return not_done

def is_terminal_region_for_bulletHP(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
    if isinstance(state, np.ndarray):
        height = state[..., 0:1]
        not_done = np.ones_like(height)
    elif isinstance(state, torch.Tensor):
        height = state[..., 0:1]
        not_done = torch.ones_like(height)
    else:
        raise ValueError
        
    if return_done:
        done = ~not_done
        return done
    else:
        return not_done


def is_terminal_region_for_bulletHC(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
    if isinstance(state, np.ndarray):
        height = state[..., 0:1]
        not_done = np.ones_like(height)
    elif isinstance(state, torch.Tensor):
        height = state[..., 0:1]
        not_done = torch.ones_like(height)
    else:
        raise ValueError
        
    if return_done:
        done = ~not_done
        return done
    else:
        return not_done


def call_terminal_func(env_name: str) -> Callable:
    if "Hopper" in env_name:        
        return is_terminal_region_for_hp
    elif "HalfCheetah" in env_name:
        return is_terminal_region_for_hc
    elif "Ant" in env_name:
        return is_terminal_region_for_at
    elif "Swimmer" in env_name:
        return is_terminal_region_for_sw
    elif "Walker" in env_name:
        return is_terminal_region_for_wk
    elif 'Humanoid' in env_name:
        return is_terminal_region_for_hm
    elif 'BulletHP' in env_name:
        return is_terminal_region_for_bulletHP
    elif 'BulletHC' in env_name:
        return is_terminal_region_for_bulletHC
    else:
        raise NotImplementedError(f'no terminal func for env {env_name}')
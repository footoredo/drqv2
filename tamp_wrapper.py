import collections
import numpy as np

import dm_env
from dm_env import StepType, specs, TimeStep

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import robosuite.utils.transform_utils as T
from robosuite.wrappers.tamp_gated_wrapper import TAMPGatedWrapper


def _spec_from_observation(observation):
    result = collections.OrderedDict()
    for key, value in observation.items():
        result[key] = specs.Array(value.shape, value.dtype, name=key)
    return result


class TAMPWrapper(dm_env.Environment):
    def __init__(self, env_name, controller="OSC_POSE"):
        options = {}
        options["env_name"] = env_name
        if "TwoArm" in options["env_name"]:
            # Choose env config and add it to options
            options["env_configuration"] = choose_multi_arm_config()

            # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
            if options["env_configuration"] == 'bimanual':
                options["robots"] = 'Baxter'
            else:
                options["robots"] = []

                # Have user choose two robots
                print("A multiple single-arm configuration was chosen.\n")

                for i in range(2):
                    print("Please choose Robot {}...\n".format(i))
                    options["robots"].append(choose_robots(exclude_bimanual=True))

        # Else, we simply choose a single (single-armed) robot to instantiate in the environment
        else:
            # options["robots"] = choose_robots(exclude_bimanual=True)
            options["robots"] = "Panda"
            
        options["controller_configs"] = load_controller_config(default_controller=controller)
        
        env = suite.make(
            **options,
            has_renderer=False,
            # has_offscreen_renderer=True,
            # ignore_done=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=84,
            camera_widths=84,
            control_freq=20,
        )
        
        # wrap with TAMP-gated wrapper
        env = TAMPGatedWrapper(
            env=env,
            htamp_grasp_conditions=False,
            htamp_constraints_path=None,
            htamp_noise_level=0,
        )
        
        self._env = env
        self._image = None
        self._num_step = None
        
    def observation_spec(self):
        return _spec_from_observation(self._env.observation_spec())
    
    def action_spec(self):
        low, high = self._env.action_spec
        # print(low.shape, high.shape)
        # return specs.BoundedArray(low.shape, low.dtype, minimum=low, maximum=high)
        return specs.BoundedArray((7,), low.dtype, minimum=low[:7], maximum=high[:7])
    
    def set_use_tamp(self, value):
        self._env.set_use_tamp(value)
        
    def reset(self) -> TimeStep:
        obs = self._env.reset()
        self._image = obs["agentview_image"].copy()
        self._num_step = 0
        
        return TimeStep(
            step_type=StepType.FIRST,
            reward=None,
            discount=None,
            observation=obs
        )
        
    def step(self, action) -> TimeStep:
        obs, reward, done, _ = self._env.step(action)
        self._image = obs["agentview_image"].copy()
        self._num_step += 1
        # print(self._num_step, reward, done)
        if not done:
            return TimeStep(
                step_type=StepType.MID,
                reward=reward,
                discount=1.0,
                observation=obs
            )
        else:
            return TimeStep(
                step_type=StepType.LAST,
                reward=reward,
                discount=0.0,
                observation=obs
            )
            
    def render(self):
        return self._image

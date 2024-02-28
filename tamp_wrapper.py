import collections
import multiprocessing as mp
import numpy as np
import joblib
import h5py

import dm_env
from dm_env import StepType, specs, TimeStep

import robosuite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import robosuite.utils.transform_utils as T
from robosuite.wrappers.tamp_gated_wrapper import TAMPGatedWrapper
from robosuite.wrappers import Wrapper


def _spec_from_observation(observation):
    result = collections.OrderedDict()
    for key, value in observation.items():
        result[key] = specs.Array(value.shape, value.dtype, name=key)
    return result


class StateResetWrapper(Wrapper):
    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        return dict(model=xml, states=state)

    def reset(self, init_state=None):
        if init_state is None:
            obs = self.env.reset()
        else:
            if "model" in init_state:
                self.env.reset()
                robosuite_version_id = int(robosuite.__version__.split(".")[1])
                if robosuite_version_id <= 3:
                    from robosuite.utils.mjcf_utils import postprocess_model_xml
                    xml = postprocess_model_xml(init_state["model"])
                else:
                    # v1.4 and above use the class-based edit_model_xml function
                    xml = self.env.edit_model_xml(init_state["model"])
                self.env.reset_from_xml_string(xml)
                self.env.sim.reset()
            if "states" in init_state:
                self.env.sim.set_state_from_flattened(init_state["states"])
                self.env.sim.forward()
            
            obs = self._get_observations(force_update=True)
        
        return obs


class TAMPWrapper(dm_env.Environment):
    def __init__(self, env_name, controller="OSC_POSE", camera=True, state_queue: mp.Queue = None):
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
        
        # self._camera_name = "agentview"
        self._camera_name = "robot0_eye_in_hand"
        
        if camera:
            camera_options = dict(
                use_camera_obs=True,
                camera_names=self._camera_name,
                camera_heights=84,
                camera_widths=84,
            )
        else:
            camera_options = dict(
                has_offscreen_renderer=False,
                use_camera_obs=False,
            )
            
        env = suite.make(
            **options,
            **camera_options,
            has_renderer=False,
            horizon=1000,
            control_freq=20,
            reward_shaping=False,
            hard_reset=False
        )
            
        env = StateResetWrapper(env)
        
        if state_queue is None:
            # wrap with TAMP-gated wrapper
            env = TAMPGatedWrapper(
                env=env,
                htamp_grasp_conditions=False,
                htamp_constraints_path=None,
                htamp_noise_level=0,
            )
            
            env.set_use_tamp(False)  # for benchmarking
        
        self._env = env
        self._image = None
        self._num_step = None
        self._state_queue = state_queue
        self._init_states = collections.deque(maxlen=100)
        
        # self._init_states = joblib.load("/home/zihanz/playground/drqv2/Stack_init_states_100.joblib")
        # self.set_use_tamp(False)
        
        f = h5py.File("/home/zihanz/playground/robomimic/datasets/stack/hmat_demo_human.hdf5", "r")
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        
        eef_hmats = []
        cubeA_hmats = []
        cubeB_hmats = []
        
        for ind in range(len(demos)):
            ep = demos[ind]
            eef_hmats.append(f[f"data/{ep}/ee"])
            cubeA_hmats.append(f[f"data/{ep}/cubeA"])
            cubeB_hmats.append(f[f"data/{ep}/cubeB"])
            
        self._eef_hmats = eef_hmats
        self._cubeA_hmats = cubeA_hmats
        self._cubeB_hmats = cubeB_hmats
        self._init_obj_hmat = None
        
    @property
    def pixels_key(self):
        return self._camera_name + "_image"
        
    def observation_spec(self):
        return _spec_from_observation(self._env.observation_spec())
    
    def action_spec(self):
        low, high = self._env.action_spec
        # print(low.shape, high.shape)
        # return specs.BoundedArray(low.shape, low.dtype, minimum=low, maximum=high)
        return specs.BoundedArray((7,), low.dtype, minimum=low[:7], maximum=high[:7])
    
    def set_use_tamp(self, value):
        self._env.set_use_tamp(value)
        
    def generate_init_state(self):
        orig_use_tamp = self._env.get_use_tamp()
        self.set_use_tamp(True)
        self._env.reset()
        state = self._env.get_state()
        self.set_use_tamp(orig_use_tamp)
        return state
    
    def get_eef_hmat(self):
        eef_site_name = self._env.robots[0].controller.eef_name
        eef_pos = np.array(self._env.sim.data.site_xpos[self._env.sim.model.site_name2id(eef_site_name)])
        eef_rot = np.array(self._env.sim.data.site_xmat[self._env.sim.model.site_name2id(eef_site_name)].reshape([3, 3]))
        eef_pose = np.zeros((4, 4)) # eef pose in world frame
        eef_pose[:3, :3] = eef_rot
        eef_pose[:3, 3] = eef_pos
        eef_pose[3, 3] = 1.0
        return eef_pose
    
    def get_obj_hmat(self):
        return self._env.get_hmat(self._env.cubeB_body_id)
    
    def _get_intrinsic_reward(self, ind):
        demo_eef = self._eef_hmats[ind]
        demo_obj = self._cubeB_hmats[ind]
        cur_eef = self.get_eef_hmat()
        trans = demo_obj[0] @ np.linalg.inv(self._init_obj_hmat)
        dists = []
        for i in range(len(demo_eef)):
            target_eef = trans @ demo_eef[i]
            dists.append(np.linalg.norm(target_eef[:3, 3] - cur_eef[:3, 3]))
        anchor = np.argmin(dists)
        dist = dists[min(anchor + 1, len(dists) - 1)]
        seg_dist_rew = 1. - np.tanh(dist)
        return (anchor + seg_dist_rew) / len(dists)
    
    def get_intrinsic_reward(self):
        return self._get_intrinsic_reward(0)
    
    def get_init_state(self):
        state_available = self._state_queue is not None and not self._state_queue.empty()
        if not state_available and len(self._init_states) > 0:
            print("Using cached initial state!")
            state = np.random.choice(self._init_states)  # TODO: seeding
        elif self._state_queue is not None:
            print("Prepared to retrieve new initial state!")
            state = self._state_queue.get()
            print("Retrieved new initial state!")
            self._init_states.append(state)  # TODO: reservoir sampling
        else:
            raise Exception("No state queue to retrieve init states")
        return state
        
    def reset(self) -> TimeStep:
        if self._state_queue is None:
            obs = self._env.reset()
        else:
            init_state = self.get_init_state()
            obs = self._env.reset(init_state=init_state)
        # self._image = obs["agentview_image"].copy()
        self._num_step = 0
        self._init_obj_hmat = self.get_obj_hmat()
        
        return TimeStep(
            step_type=StepType.FIRST,
            reward=None,
            discount=None,
            observation=obs
        )
        
    def step(self, action) -> TimeStep:
        obs, reward, done, _ = self._env.step(action)
        # if reward > 0.5:
        #     done = True
        reward += self.get_intrinsic_reward() * 0.01
        # self._image = obs["agentview_image"].copy()
        self._num_step += 1
        if self._num_step >= 200:
            done = True
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
        return self._env.sim.render(height=512, width=512, camera_name="agentview")[::-1]

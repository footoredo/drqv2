import numpy as np
import time

from dm_env import StepType, specs, TimeStep
from tamp_wrapper import TAMPWrapper

env = TAMPWrapper("Stack")

low, high = env._env.action_spec
env.set_use_tamp(False)

env.reset()

st = time.time()

for i in range(10000):
    action = np.random.uniform(low, high)
    ts = env.step(action)
    if (i + 1) % 100 == 0:
        print(f"Timestep {i + 1}, fps={(i + 1) / (time.time() - st):.3f}")
    if ts.step_type == StepType.LAST:
        print("Done!")
        env.reset()
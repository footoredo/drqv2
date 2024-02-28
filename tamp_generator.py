import os
import sys
import time
import queue
from pathlib import Path
import multiprocessing as mp
import numpy as np

from dm_env import StepType, specs, TimeStep
from tamp_wrapper import TAMPWrapper


class TAMPStateGenerator(mp.Process):
    def __init__(self, gid, env_args, queue: mp.Queue, halt_queue: mp.Queue, work_dir):
        self.gid = gid
        self.env_args = env_args
        self.queue = queue
        self.halt_queue = halt_queue
        if work_dir is not None:
            self.log_file = Path(work_dir) / f"TSG-{gid}.out"
        else:
            self.log_file = None
        super().__init__(daemon=True)
    
    def get_pid(self):
        print("PID: ",os.getpid())
        
    def run(self):
        self.get_pid()
        env = TAMPWrapper(camera=False, **self.env_args)
        init_state = None
        if self.log_file is not None:
            sys.stdout = open(str(self.log_file), "w")
        while self.halt_queue.empty():
            if init_state is None:
                init_state = env.generate_init_state()
            try:
                self.queue.put(init_state, block=False)
                init_state = None
                # return
            except queue.Full:
                time.sleep(1)
            

class TAMPStateGeneratorController:
    def __init__(self, env_args, num_process, maxsize, work_dir=None):
        self.q = mp.Queue(maxsize=maxsize)
        self.hq = mp.Queue()
        self.ps = []
        self.work_dir = work_dir
        for i in range(num_process):
            p = TAMPStateGenerator(i, env_args, self.q, self.hq, self.work_dir)
            self.ps.append(p)
        
    def start(self):
        for p in self.ps:
            p.start()
            
    def stop(self):
        self.hq.put(None)
        # print("put")
        for p in self.ps:
            p.join()


def test():
    con = TAMPStateGeneratorController(dict(env_name="Stack"), 1, 1, work_dir="/home/zihanz/playground/drqv2")
    env = TAMPWrapper(env_name="Stack", state_queue=con.q)
    con.start()
    # env.reset()
    # for _ in range(10):
    #     print("qsize:", con.q.qsize())
    #     time.sleep(1)py
    env.reset()
    st = time.time()
    low = env.action_spec().minimum
    high = env.action_spec().maximum
    last_reset = -1
    for i in range(1000):
        action = np.random.uniform(low, high)
        # action = np.zeros_like(action)
        action[-1] = 1.
        obs, reward, done, _ = env.step(action)
        if i - last_reset > 100:
            env.reset()
            last_reset = i
        print("fps:", (i + 1) / (time.time() - st))
        
    con.stop()
    

if __name__ == "__main__":
    test()

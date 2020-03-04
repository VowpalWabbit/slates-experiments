from multi_d_simulator import *

class Solution():
    
    @staticmethod
    def gen_trajectory(sim_summary):
        trajectory = MultiDSimulator.gen_trajectory(sim_summary, 1000, include_sample_size=True, include_reward=False)
        return trajectory
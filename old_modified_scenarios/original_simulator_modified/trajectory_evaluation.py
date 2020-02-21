import sys
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

class TrajectoryEvaluation():
    
    def __init__(self, trajectory_file, summary_file, opt_reward, debug=False):
        self.trajectory_file = trajectory_file
        self.summary_file = summary_file
        self.opt_reward = opt_reward
        self.debug = debug
        
    def read_trajectory(self):
        df_trajectory = pd.read_csv(self.trajectory_file, header=None)
        if self.debug:
            if df_trajectory.shape[1]==2:
                df_trajectory.columns = ['context', 'config']
            if df_trajectory.shape[1]==3:
                df_trajectory.columns = ['context', 'config', 'sample_size']
            elif df_trajectory.shape[1]==4:
                df_trajectory.columns = ['context', 'config', 'sample_size', 'reward']
            else:
                raise ValueError('Invalid trajectory format.')
        else:
            if df_trajectory.shape[1]==3:
                df_trajectory.columns = ['context', 'config', 'sample_size']
            else:
                raise ValueError('Invalid trajectory format. Each line should be in the format of "[contexts]", "(config)", sample_size')
        return df_trajectory

    def complete_trajectory(self, df_trajectory, df_summary):
        if 'sample_size' not in df_trajectory.columns:
            df_trajectory['sample_size'] = 1
        if 'reward' not in df_trajectory.columns:
            df_trajectory = self.find_nearest_reward(df_trajectory, df_summary)
        return df_trajectory

    def find_nearest_reward(self, df_trajectory, df_summary):
        df_trajectory_completed = pd.DataFrame()
        for c in df_trajectory['context'].unique():
            # All config summary
            df_summary_context = df_summary.loc[df_summary['context']==c].reset_index(drop=True).copy()           
            array_grids = np.array([eval(x) for x in df_summary_context['config'].values])
            # Trajectory configs
            df_trajectory_context = df_trajectory.loc[df_trajectory['context']==c].copy()
            array_config = np.array([eval(x) for x in df_trajectory_context['config'].values])
            # Neareast from the config summary
            dist = cdist(array_config, array_grids)
            df_trajectory_context['reward'] = df_summary_context.loc[dist.argmin(1), 'reward'].values
            df_trajectory_completed = df_trajectory_completed.append(df_trajectory_context)
        return df_trajectory_completed

    def optimal_reward(self, df_summary):
        df_opt = df_summary.groupby('context').agg({'reward': self.opt_reward}).reset_index()
        df_opt = pd.merge(df_opt, df_summary, how='left', left_on=['context', 'reward'], right_on=['context', 'reward'])
        return df_opt
    
    def prep_data(self):
        df_summary = pd.read_csv(self.summary_file)
        df_opt = self.optimal_reward(df_summary)
        df_trajectory = self.read_trajectory()
        df_trajectory = self.complete_trajectory(df_trajectory, df_summary)
        df = pd.merge(df_trajectory, df_opt, how='left', left_on=['context'], right_on=['context'], suffixes=['', '_opt'])
        df = self.add_regret(df)
        df['reward_total'] = df['reward']*df['sample_size']
        df['regret_total'] = df['regret']*df['sample_size']
        return df
    
    def add_regret(self, df):
        if self.opt_reward == 'min':
            df['reward'] = -1.0*df['reward']
            df['reward_opt'] = -1.0*df['reward_opt']
        df['regret'] = df['reward'] - df['reward_opt']
        return df
    
    def agg_df(self, df_group):
        agg = {}
        agg['Total_N'] = df_group['sample_size'].sum()
        agg['Optimal_Reward'] = df_group['reward_opt'].mean()
        df_last5 = df_group.loc[df_group.loc[::-1, 'sample_size'].cumsum()[::-1]<=max(5, df_group['sample_size'].values[-1])]
        agg['Last_5_Rewards_Avg'] = df_last5['reward_total'].sum()/df_last5['sample_size'].sum()
        agg['Diff_from_Optimal'] = str(round(((agg['Last_5_Rewards_Avg']+1e-10)/(agg['Optimal_Reward']+1e-10)-1)*100, 1)) + '%'
        agg['Total_Regret'] = df_group['regret_total'].sum()
        agg['Avg_Regret'] = agg['Total_Regret']/agg['Total_N']
        s_agg = pd.Series(agg)
        return s_agg
    
    def notes(self):
        notes = ''' Notes:
        * Total_N: Total number of samples explored.
        * Optimal_Reward: The best average reward from the ground truth file.
        * Last_5_Rewards_Avg: The average reward from the last 5 samples in a trajectory
        * Diff_from_Optimal: The difference between Last_5_Rewards_Avg and Optimal_Reward in terms of %.
        * Total_Regret: The sum of the regrets in a trajectory
        * Avg_Regret: The average regrets for each sample in a trajectory
        '''
        return notes    
    
    def evaluate(self):
        df = self.prep_data()
        df_summary = df.groupby('context').apply(lambda group: self.agg_df(group))
        df_summary = df_summary.round(4)
        return df_summary
    

if __name__ == "__main__":
    '''
    Inputs:
    trajectory_file: path to the trajectory file. The file should be comma separated. Each line in the format of "[context]", "(configuration)", sample_size. 
                     eg: "['Windows', 'wired', 'CA']","(3.79, 0.11, 1.05)",8
                     In debug mode, you can also pass a reward as the 4th element. 
                     If empty, sample_size will be filled with 1 for each configuration while reward will be the average reward from the nearest configuration according to the ground truth summary file.
    summary_file: path to the ground truth summary file.
    opt_reward: min or max
    '''
    te = TrajectoryEvaluation(sys.argv[1], sys.argv[2], sys.argv[3], debug=False)
    df_summary = te.evaluate()
    print(df_summary)
    print(te.notes())
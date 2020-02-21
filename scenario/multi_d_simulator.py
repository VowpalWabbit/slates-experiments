import copy
import itertools
import os
import pandas as pd
import numpy as np
np.random.seed(7)

import matplotlib.pyplot as plt
import scipy.stats as sts

class MultiDSimulator:
    """
    Generate simulated datasets for the multi-d scenario.
    """

    def __init__(self, **kwargs):
        self.folder_path = kwargs['folder_path']
        self.contexts = kwargs['contexts']
        self.actions = kwargs['actions']
        self.param_list = list(self.actions.keys())
        self.discretization_fine_grain = kwargs.get('discretization_fine_grain', 100)
        self.discretization_policy = kwargs['discretization_policy']
        self.share_discretized_grid = kwargs.get('share_discretized_grid', True)
        self.reward_range = kwargs['reward_range']
        self.reward_minimization = kwargs['reward_minimization']
        self.interaction_level = kwargs.get('interaction_level', len(self.actions))
        self.coefficient_range = kwargs.get('coefficient_range', [0.1, 2])
        self.coefficient_scale_range = kwargs.get('coefficient_scale_range', [0.8, 1.2])
        self.dist_mean_change_range = kwargs.get('dist_mean_change_range', [0.6, 1.4])
        self.dist_std_change_range = kwargs.get('dist_std_change_range', [0.6, 1.4])
        self.known_n_per_config = kwargs.get('known_n_per_config', 10)
        self.ci_mean = kwargs.get('ci_mean', 0)
        self.ci_std = kwargs.get('ci_std', 0.01)
        self.ci_width = kwargs.get('ci_width', (self.reward_range[1]-self.reward_range[0])/50)
        self.update_args()
        self.summarize_task()
        
    def update_args(self):
        self.summary_file_path = os.path.join(self.folder_path, 'simulation_data_summary.csv')
        self.context_file_path = os.path.join(self.folder_path, 'simulation_data_{0}.csv')        
        self.unique_contexts = [list(x) for x in itertools.product(*self.contexts.values())]
        self.update_discretization_policy(self.discretization_policy, self.actions, self.share_discretized_grid)
        self.opt_target = 'Cost' if self.reward_minimization else 'Reward'
        self.ci_dist = self.gen_distribution('normal', self.ci_mean, self.ci_std, 5000)[0]
        self.n_per_config = self.get_n(self.known_n_per_config, self.ci_std, self.ci_width)
        self.nunique_configs = np.prod([x if isinstance(x, int) else len(x)for x in self.discretization_policy.values()]) 
        self.n_per_context = self.n_per_config * self.nunique_configs
        self.plot_pairs = [x for x in itertools.combinations(range(len(self.param_list)), 2)]
        self.max_discretization = max([x if isinstance(x, int) else len(x)for x in self.discretization_policy.values()])
        self.discretization_base = {k: max(self.discretization_fine_grain, self.max_discretization) for k in self.param_list}
        self.df_cols = list(self.contexts.keys()) + self.param_list + ['reward']

    def summarize_task(self):
        print('='*10, 'Summary of the Simulation Task', '='*10)
        print('Data Size per Configuration: {:,}'.format(self.n_per_config))
        print('Numer of Unique Configurations: {:,}'.format(self.nunique_configs))
        print('Data Size per Context: {:,}'.format(self.n_per_context))
        print('Numer of Unique Contexts: {:,}'.format(len(self.unique_contexts)))
        print('Total Data Size: {:,}'.format(self.n_per_context*len(self.unique_contexts)))        
        print('='*52)

    def update_discretization_policy(self, discretization_policy, actions, share_discretized_grid):
        if share_discretized_grid:
            for k, v in discretization_policy.items():
                if isinstance(v, int):
                    discretization_policy[k] = [round(x,4) for x in np.linspace(actions[k]['min'], actions[k]['max'], v)]        

    def gen_distribution(self, dist_type, mu, std, n, reverse=None, pmin=None, pmax=None, show_plot=False):
        if dist_type == 'normal':
            x = np.random.normal(mu, std, n)
            pmin = pmin or min(x)
            pmax = pmax or max(x)
            x_tick = np.linspace(pmin, pmax, n)
            x_pdf = sts.norm.pdf(x_tick, mu, std)
        elif dist_type == 'gamma':
            shape = (mu/std)**2
            scale = mu/shape
            x = np.random.gamma(shape, scale, n)
            pmin = pmin if pmin is not None else min(x)
            pmax = pmax if pmax is not None else max(x)
            x_tick = np.linspace(pmin, pmax, n)
            x_pdf = sts.gamma.pdf(x_tick, a=shape, scale=scale)
        else:
            raise ValueError('dist_type must be in ["normal", "gamma"]')
        if reverse is None:
            reverse = np.random.randint(0, 2)
        if reverse==1:
            x = pmax - x + pmin
            x_pdf = x_pdf[::-1]
        if show_plot:
            plt.hist(x, bins=25, density=True, alpha=0.6, color='darkcyan')
            plt.plot(x_tick, x_pdf, 'black', linewidth=2)
            plt.xlim((pmin, pmax))
            plt.show()
        return x, x_tick, x_pdf, [dist_type, mu, std, n, reverse, pmin, pmax]

    def get_n(self, known_n, ci_std, ci_width, ci_mult=1.96):
        if known_n:
            n_per_config = known_n
        else:
            n_per_config = int(((ci_mult*ci_std/(ci_width/2))**2//10+1)*10)
        return n_per_config

    def gen_param_reward(self, n_dist=5000, plot=True):
        n_dist = max(n_dist, self.max_discretization)
        param_reward = {}
        for p, pv in self.actions.items():
            pmu, pmin, pmax = pv['mean'], pv['min'], pv['max']
            pstd = np.random.uniform(min(0.1, pv['std_range'][0]), min(pv['mean'], pv['std_range'][1]))
            param_reward[p] = {}
            param_reward[p]['raw'], param_reward[p]['tick'], param_reward[p]['pdf'], param_reward[p]['dist_inputs'] \
                = self.gen_distribution('gamma', pmu, pstd, n_dist, reverse=None, pmin=pmin, pmax=pmax)
        for p in self.actions.keys():
            param_reward[p]['pdf'] = self.rescale_reward(param_reward[p]['pdf'], [0, 1])
            param_reward[p]['pdf'] = 1 - param_reward[p]['pdf']
            param_reward[p]['pdf'] = self.rescale_reward(param_reward[p]['pdf'], self.reward_range)
        if plot:
            self.plot_1d_param_reward(param_reward)    
        return param_reward

    def gen_coefficients(self):
        inter_terms = []
        for i in range(1, self.interaction_level):
            inter_terms = inter_terms + [x for x in itertools.combinations(range(len(self.param_list)), i+1)]
        n_coef = len(self.param_list) + len(inter_terms)
        coefficients = np.random.uniform(self.coefficient_range[0], self.coefficient_range[1], n_coef)    
        return inter_terms, coefficients

    def plot_1d_param_reward(self, param_dist):
        plt.figure(figsize=(15,2))
        pn = len(self.param_list)
        i = 1
        for p, pv in param_dist.items():
            if p not in self.param_list:
                continue
            subpn = int('1{0}{1}'.format(pn, i))
            ax = plt.subplot(subpn)
            ax.plot(list(pv['tick']), list(pv['pdf']), 'black', linewidth=2)
            ax.fill_between(list(pv['tick']), 0, list(pv['pdf']), facecolor='grey', alpha=0.3)
            ax.set_title(p)
            ax.set_xlabel('parameter value')
            ax.set_ylabel(self.opt_target)
            i = i +1
        plt.show()

    def discretize(self, dist, discretization_policy=None, coefficients=None):
        if discretization_policy is None:
            discretization_policy = self.discretization_base
        if coefficients is None:
            self.inter_terms, self.coefficients_base  = self.gen_coefficients()
            coefficients = self.coefficients_base 
        self.discretize_parameters(dist, discretization_policy)
        self.reward_formula = ['f<sub>{0}</sub>({0})'.format(p) for p in self.param_list] + [''.join(['f<sub>{0}</sub>({0})'.format(self.param_list[y]) for y in x]) for x in self.inter_terms]
        dist['configs'] = self.gen_config_reward(dist)
        dist['configs']['reward_equation'] = self.formulate_equation(coefficients)
        dist['configs']['config_rterms'] = self.add_interactions(dist['configs']['config_rterms'])

    def discretize_parameters(self, dist, discretization_policy, equal_distance=True):
        for k in self.param_list:
            n = discretization_policy[k]
            if isinstance(n, int):
                if equal_distance:
                    grid_idx = np.linspace(0, len(dist[k]['tick'])-1, n).astype(int)
                    dist[k]['grid'] = np.array([dist[k]['tick'][x] for x in grid_idx])
                else:
                    dist[k]['grid'] = np.sort(np.random.choice(dist[k]['tick'], n, replace=False))
                dist[k]['grid_reward'] = np.array([dist[k]['pdf'][np.where(dist[k]['tick'] == x)][0] for x in dist[k]['grid']])
            elif isinstance(n, list):
                dist[k]['grid'] = np.array(n)
                grid_idx = [np.argmin(abs(dist[k]['tick']-x)) for x in dist[k]['grid']]
                nearest_tick = np.array([dist[k]['tick'][x] for x in grid_idx])
                dist[k]['grid_reward'] = np.array([dist[k]['pdf'][np.where(dist[k]['tick'] == x)][0] for x in nearest_tick])
            else:
                raise TypeError("Discretization policy can either be an int or a list.")

    def gen_config_reward(self, dist):
        config_reward = {}
        # All configurations by index
        grids_length = [range(len(dist[x]['grid'])) for x in self.param_list]
        config_reward['config_idx'] = np.array([list(x) for x in itertools.product(*grids_length)])
        # All configurations (parameter values)
        config_reward['config_val'] = np.array([dist[p]['grid'][config_reward['config_idx'][:,i]] for i, p in enumerate(self.param_list)]).T
        # All configurations' rewards
        config_reward['config_rterms'] = np.array([dist[p]['grid_reward'][config_reward['config_idx'][:,i]] for i, p in enumerate(self.param_list)]).T
        return config_reward

    def formulate_equation(self, coefficients):
        equation = '{0} = {1}'.format(self.opt_target, ' + '.join(['{0}\*{1}'.format(round(coefficients[i], 4), self.reward_formula[i]) for i in range(len(coefficients))]))
        return equation

    def add_interactions(self, reward_terms):
        for t in self.inter_terms:
            t_v = np.prod(reward_terms[:, list(t)], axis=1)
            reward_terms = np.append(reward_terms, t_v.reshape(len(t_v), 1), 1)
        return reward_terms

    def gen_data(self, dist, n, coefficients=None, add_error=True, data_min=None, data_max=None, plot_2d=False):
        coefficients = self.coefficients_base if coefficients is None else coefficients
        rterms = np.tile(dist['configs']['config_rterms'], (n, 1))
        par_values = np.tile(dist['configs']['config_val'], (n, 1))
        reward_total, reward_raw_min, reward_raw_max = self.combine_elements(rterms, coefficients, data_min, data_max)
        if add_error:
            errors = np.random.choice(self.ci_dist, len(rterms), replace=True)
            reward_total = np.sum((reward_total, errors), axis=0)
        num_values = np.concatenate((par_values, reward_total.reshape(-1,1)), axis=1)
        dist['configs']['config_reward'] = np.array([x[-1] for x in num_values])
        if plot_2d:
            plot_data = pd.DataFrame(num_values, columns=self.param_list+['reward'])
            self.plot_2d_paris(plot_data)
        return num_values, reward_raw_min, reward_raw_max

    def combine_elements(self, reward_terms, coefficients, data_min=None, data_max=None):
        reward_terms = np.multiply(reward_terms, coefficients)
        reward_sum = np.sum(reward_terms, 1)
        reward_min = reward_sum.min()
        reward_max = reward_sum.max()
        reward_sum_rescale = self.rescale_reward(reward_sum, self.reward_range, data_min, data_max)
        return reward_sum_rescale, reward_min, reward_max

    def rescale_reward(self, s, reward_scale, data_min=None, data_max=None):
        min_noninf = np.min([x for x in s if x != -np.inf])
        max_noninf = np.max([x for x in s if x != np.inf])
        s = np.nan_to_num(s, posinf=max_noninf, neginf=min_noninf)
        data_min = min_noninf if data_min is None else data_min
        data_max = max_noninf if data_max is None else data_max
        reward_rescale = (s-data_min)/(data_max-data_min) * (reward_scale[1]-reward_scale[0]) + reward_scale[0]
        return reward_rescale

    def plot_2d_paris(self, plot_data, round_to=0.01, cmap='viridis_r'):
        plt.figure(figsize=(15,3))
        for p in self.param_list:
            plot_data[p] = plot_data[p]//round_to*round_to
        for i, t in enumerate(self.plot_pairs):
            df_grid = plot_data.groupby([self.param_list[t[0]], self.param_list[t[1]]]).agg({'reward': 'mean'}).unstack(0)
            df_grid.columns = df_grid.columns.droplevel(0)
            df_grid.fillna(method='ffill', inplace=True)
            subpn = int('1{0}{1}'.format(len(self.plot_pairs), i+1))
            ax = plt.subplot(subpn)    
            ax.pcolor(df_grid.columns, df_grid.index, df_grid, cmap=cmap)
            ax.set_xlabel(df_grid.columns.name)
            ax.set_ylabel(df_grid.index.name)
            ax.set_title('{0} on {1} vs {2}'.format(self.opt_target, df_grid.columns.name, df_grid.index.name))
        plt.show()
        
    def random_changes(self):
        context_dist_change = {}
        for values in self.contexts.values():
            for v in values:
                v_mean_scale = np.random.uniform(self.dist_mean_change_range[0], self.dist_mean_change_range[1], len(self.param_list)) 
                v_std_scale = np.random.uniform(self.dist_std_change_range[0], self.dist_std_change_range[1], len(self.param_list)) 
                v_coeff_scale = np.random.uniform(self.coefficient_scale_range[0], self.coefficient_scale_range[1], len(self.coefficients_base))
                context_dist_change[v] = {'mean_scale': v_mean_scale, 'std_scale': v_std_scale, 'coeff_scale': v_coeff_scale}
        self.context_dist_change = context_dist_change

    def adjust_distributuion(self, dist_context, dist_base, context, plot=True):
        c_name = '_'.join(context)
        dist_context[c_name] = copy.deepcopy(dist_base)
        for i, p in enumerate(self.param_list):
            dist_in = dist_base[p]['dist_inputs']
            p_mean =  dist_in[1]
            p_std =  dist_in[2]
            for f in context:
                p_mean = p_mean * self.context_dist_change[f]['mean_scale'][i]
                p_std = p_std * self.context_dist_change[f]['std_scale'][i]
            tmp = {}
            tmp['raw'], tmp['tick'], tmp['pdf'], tmp['dist_inputs'] = self.gen_distribution(dist_in[0], p_mean, p_std, dist_in[3], dist_in[4], dist_in[5], dist_in[6])
            tmp['pdf'] = self.rescale_reward(tmp['pdf'], [0, 1])
            tmp['pdf'] = 1 - tmp['pdf']
            tmp['pdf'] = self.rescale_reward(tmp['pdf'], self.reward_range)
            dist_context[c_name][p] = copy.deepcopy(tmp)
        if plot:
            self.plot_1d_param_reward(dist_context[c_name])

    def adjust_coefficients(self, context):
        c_coeff = self.coefficients_base.copy()
        for f in context:
            c_coeff = c_coeff*self.context_dist_change[f]['coeff_scale']
        return c_coeff

    def export_data(self, context, data, to_csv=False):
        c_name = '_'.join(context)
        c_data = [context + list(x) for x in data]
        df_context = pd.DataFrame(c_data, columns=self.df_cols)
        df_context = df_context.sample(frac=1)
        if to_csv:
            df_context.to_csv(self.context_file_path.format(c_name), index=False)
        return df_context

    def summarize_df(self, df_summary, context, num_values):
        gt_data = [context + list(x) for x in num_values]
        df_context = pd.DataFrame(gt_data, columns=self.df_cols)    
        df_mean = df_context.groupby(self.param_list).agg({'reward': 'mean'})
        df_mean['config'] = df_mean.index.values
        df_mean['context'] = str(context)
        df_mean.reset_index(inplace=True, drop=True)
        df_summary = df_summary.append(df_mean)
        return df_summary

    def gen_trajectory(self, df_summary, length, include_sample_size=True, sample_size=1, inclue_reward=True):
        ss = df_summary.sample(length).copy()
        to_keep = ['context', 'config']
        if include_sample_size:
            ss['sample_size'] = sample_size
            to_keep = to_keep + ['sample_size']
        if inclue_reward:
            to_keep = to_keep + ['reward']
        return ss[to_keep]
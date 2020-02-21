import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

def gen_distribution(dist_type, mu, std, n, pmin=None, pmax=None, show_plot=False):
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
        pmin = pmin or min(x)
        pmax = pmax or max(x)
        x_tick = np.linspace(pmin, pmax, n)
        x_pdf = sts.gamma.pdf(x_tick, a=shape, scale=scale)
    else:
        raise ValueError('dist_type must be in ["normal", "gamma"]')
    reverse = np.random.randint(0, 2)
    if reverse==1:
        x = pmax - x + pmin
        x_pdf = x_pdf[::-1]
    if show_plot:
        plt.hist(x, bins=25, density=True, alpha=0.6, color='darkcyan')
        plt.plot(x_tick, x_pdf, 'black', linewidth=2)
        plt.xlim((pmin, pmax))
        plt.show()
    return x, x_tick, x_pdf

def formulate_equation(reward_formula, coefficients):
    equation = 'reward = {0}'.format(' + '.join(['{0}{1}'.format(round(coefficients[i], 4), reward_formula[i]) for i in range(len(coefficients))]))
    return equation

def add_interactions(reward_terms, inter_terms):
    for t in inter_terms:
        t_v = np.multiply(reward_terms[:,t[0]], reward_terms[:,t[1]])
        reward_terms = np.append(reward_terms, t_v.reshape(len(t_v), 1), 1)
    return reward_terms

def rescale_reward(s, reward_scale):
    reward_rescale = (s-s.min())/(s.max()-s.min()) * (reward_scale[1]-reward_scale[0]) + reward_scale[0]
    return reward_rescale
    
def combine_elements(reward_terms, coefficients, reward_range):
    reward_terms = np.multiply(reward_terms, coefficients)
    reward_sum = np.sum(reward_terms, 1)
    reward_sum_rescale = rescale_reward(reward_sum, reward_range)
    return reward_sum_rescale

def plot_1d_param_reward(param_dist):
    plt.figure(figsize=(15,2))
    pn = len(param_dist)
    i = 1
    for p, pv in param_dist.items():
        subpn = int('1{0}{1}'.format(pn, i))
        ax = plt.subplot(subpn)
        ax.plot(list(pv['tick']), list(pv['pdf']), 'black', linewidth=2)
        ax.fill_between(list(pv['tick']), 0, list(pv['pdf']), facecolor='grey', alpha=0.3)
        ax.set_title(p)
        ax.set_xlabel('parameter value')
        ax.set_ylabel('reward')
        i = i +1
    plt.show()
    
def plot_2d_paris(plot_data, param_list, inter_terms, round_to=0.1, cmap='viridis_r'):
    plt.figure(figsize=(15,3))
    for p in param_list:
        plot_data[p] = plot_data[p]//round_to*round_to
    for i, t in enumerate(inter_terms):
        df_grid = plot_data.groupby([param_list[t[0]], param_list[t[1]]]).agg({'reward': 'mean'}).unstack(0)
        df_grid.columns = df_grid.columns.droplevel(0)
        df_grid.fillna(method='ffill', inplace=True)
        subpn = int('1{0}{1}'.format(len(inter_terms), i+1))
        ax = plt.subplot(subpn)    
        ax.pcolor(df_grid.columns, df_grid.index, df_grid, cmap=cmap)
        ax.set_xlabel(df_grid.columns.name)
        ax.set_ylabel(df_grid.index.name)
        ax.set_title('Reward on {0} vs {1}'.format(df_grid.columns.name, df_grid.index.name))
    plt.show()
    
def gen_param_reward(params, reward_range, n_dist=5000):
    param_reward = {}
    for p, pv in params.items():
        pmu, pmin, pmax = pv['mean'], pv['min'], pv['max']
        pstd = np.random.uniform(pv['std_range'][0], pv['std_range'][1])
        param_reward[p] = {}
        param_reward[p]['raw'], param_reward[p]['tick'], param_reward[p]['pdf'] = gen_distribution('gamma', pmu, pstd, n_dist, pmin=pmin, pmax=pmax)
    for p in params.keys():
        param_reward[p]['pdf'] = rescale_reward(param_reward[p]['pdf'], [0, 1])
        param_reward[p]['pdf'] = 1 - param_reward[p]['pdf']
        param_reward[p]['pdf'] = rescale_reward(param_reward[p]['pdf'], reward_range)
    return param_reward

# def discretize_parameters(dist, descritization_policy, equal_distance=True):
#     for k, v in dist.items():
#         n = descritization_policy[k]
#         if equal_distance:
#             grid_idx = np.linspace(0, len(dist[k]['tick'])-1, n).astype(int)
#             dist[k]['grid'] = np.array([dist[k]['tick'][x] for x in grid_idx])
#         else:
#             dist[k]['grid'] = np.sort(np.random.choice(dist[k]['tick'], n, replace=False))
#         dist[k]['grid_reward'] = np.array([dist[k]['pdf'][np.where(dist[k]['tick'] == x)][0] for x in dist[k]['grid']])

def discretize_parameters(dist, descritization_policy, equal_distance=True):
    for k, v in dist.items():
        n = descritization_policy[k]
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
            raise TypeError("Descritization policy can either be an int or a list.")

def gen_config_reward(dist, param_list):
    config_reward = {}
    # All configurations by index
    grids_length = [range(len(dist[x]['grid'])) for x in param_list]
    config_reward['config_idx'] = np.array([list(x) for x in itertools.product(*grids_length)])
    # All configurations (parameter values)
    config_reward['config_val'] = np.array([dist[p]['grid'][config_reward['config_idx'][:,i]] for i, p in enumerate(param_list)]).T
    # All configurations' rewards
    config_reward['config_rterms'] = np.array([dist[p]['grid_reward'][config_reward['config_idx'][:,i]] for i, p in enumerate(param_list)]).T
    return config_reward

def gen_data(dist, n, ci_dist, coefficients, reward_range):
    rterms = np.tile(dist['configs']['config_rterms'], (n, 1))
    par_values = np.tile(dist['configs']['config_val'], (n, 1))
    errors = np.random.choice(ci_dist, len(rterms), replace=True)
    reward_sum_rescale = combine_elements(rterms, coefficients, reward_range)
    reward_w_error = np.sum((reward_sum_rescale, errors), axis=0)
    num_values = np.concatenate((par_values, reward_w_error.reshape(-1,1)), axis=1)
    return num_values

def summarize_df(df_context, param_list, c):
    df_mean = df_context.groupby(param_list).agg({'reward': 'mean'})
    df_mean['config'] = df_mean.index.values
    df_mean['context'] = str(tuple(c))
    df_mean.reset_index(inplace=True, drop=True)
    return df_mean

def gen_trajectory(df_summary, length, include_sample_size=True, sample_size=1, inclue_reward=True):
    ss = df_summary.sample(length).copy()
    to_keep = ['context', 'config']
    if include_sample_size:
        ss['sample_size'] = sample_size
        to_keep = to_keep + ['sample_size']
    if inclue_reward:
        to_keep = to_keep + ['reward']
    return ss[to_keep]
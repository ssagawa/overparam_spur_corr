import numpy as np

def generate_toy_data_random_projections(n, d_causal, d_spurious, p_correlation,
                                         mean_causal, var_causal, mean_spurious, var_spurious,
                                         train=True):
    # group_idx: (y, a)
    groups = [(1,1), (1,-1), (-1,1), (-1,-1)]
    n_groups = len(groups)

    y_list, a_list, x_causal_list, x_spurious_list, g_list = [], [], [], [], []
    for group_idx, (y_value, a_value) in enumerate(groups):
        if train:
            if y_value==a_value:
                n_group = int(n/2*p_correlation)
            else:
                n_group = int(n/2*(1-p_correlation))
        else:
            n_group = int(n/n_groups)

        y_list.append(np.ones(n_group)*y_value)
        a_list.append(np.ones(n_group)*a_value)
        g_list.append(np.ones(n_group)*group_idx)
        x_causal_list.append(np.random.multivariate_normal(mean=y_value*np.ones(d_causal)*mean_causal,
                                                           cov=np.eye(d_causal)*var_causal,
                                                           size=n_group))
        x_spurious_list.append(np.random.multivariate_normal(mean=a_value*np.ones(d_spurious)*mean_spurious,
                                                             cov=np.eye(d_spurious)*var_spurious,
                                                             size=n_group))

    y = np.concatenate(y_list)
    a = np.concatenate(a_list)
    g = np.concatenate(g_list)
    x_causal = np.vstack(x_causal_list)
    x_spurious = np.vstack(x_spurious_list)
    x = np.hstack([x_causal, x_spurious])
    return x, y, g, n_groups

def generate_toy_data_no_projections(n, d_noise, p_correlation,
                                     mean_causal, var_causal, mean_spurious, var_spurious,
                                     noise_type='gaussian', mean_noise=0, var_noise=1, 
                                     train=True):
    groups = [(1,1), (1,-1), (-1,1), (-1,-1)]
    n_groups = len(groups)

    y_list, a_list, x_causal_list, x_spurious_list, g_list = [], [], [], [], []
    for group_idx, (y_value, a_value) in enumerate(groups):
        if train:
            if y_value==a_value:
                n_group = int(np.round(n/2*p_correlation))
            else:
                n_group = int(np.round(n/2*(1-p_correlation)))
        else:
            n_group = int(n/n_groups)

        y_list.append(np.ones(n_group)*y_value)
        a_list.append(np.ones(n_group)*a_value)
        g_list.append(np.ones(n_group)*group_idx)
        x_causal_list.append(np.random.normal(y_value*mean_causal,
                                              var_causal**0.5,
                                              n_group).reshape(n_group,1))
        x_spurious_list.append(np.random.normal(a_value*mean_spurious,
                                                var_spurious**0.5,
                                                n_group).reshape(n_group,1))

    x_noise = np.random.multivariate_normal(mean=mean_noise*np.ones(d_noise),
                                            cov=np.eye(d_noise)*var_noise/d_noise,
                                            size=n)
    y = np.concatenate(y_list)
    a = np.concatenate(a_list)
    g = np.concatenate(g_list)
    x_causal = np.vstack(x_causal_list)
    x_spurious = np.vstack(x_spurious_list)
    x = np.hstack([x_causal, x_spurious, x_noise])
    return (x, y, g), n_groups

def generate_toy_data(data_generation_fn, data_args):
    train_x, train_y, train_g, n_groups = data_generation_fn(**data_args, train=True)
    test_x, test_y, test_g, _ = data_generation_fn(**data_args, train=False)
    full_data = (train_x, train_y, train_g), (test_x, test_y, test_g)
    return full_data, n_groups

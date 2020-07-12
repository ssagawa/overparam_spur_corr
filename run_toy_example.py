import numpy as np
import argparse
from toy_example_data import *
from random_feature_utils import *

def main():
    parser = argparse.ArgumentParser()
    # Toy example data
    parser.add_argument('-n', '--n', type=int, required=True)
    parser.add_argument('--toy_example_name', choices=['random_projections','no_projections'], required=True)
    # Data generation parameters
    parser.add_argument('--p_correlation', type=float)
    parser.add_argument('--mean_causal', type=float)
    parser.add_argument('--mean_spurious', type=float)
    parser.add_argument('--var_causal', type=float)
    parser.add_argument('--d_causal', type=int)
    parser.add_argument('--d_spurious', type=int)
    parser.add_argument('--var_spurious', type=float)
    parser.add_argument('--d_noise', type=int)
    parser.add_argument('--mean_noise', type=float)
    parser.add_argument('--var_noise', type=float)
    # Random features
    parser.add_argument('-N', '--n_random_features', type=int, action='append')
    # Model
    parser.add_argument('-m', '--model_type', choices=['ridge','logistic'], required=True)
    parser.add_argument('-L', '--Lambda', type=float, default=None)
    # Error
    parser.add_argument('-e', '--error_type', choices=['zero_one', 'squared'], required=True)
    # Outputs
    parser.add_argument('-o', '--outfile', required=True)
    parser.add_argument('--model_file')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    process_args(args)


    if args.random_features:
        full_data, n_groups = generate_toy_data(args.data_generation_fn, args.data_args)
        erm_error, over_error, under_error = run_random_features_model(full_data=full_data,
            n_groups=n_groups,
            N=args.n_random_features,
            fit_model_fn=args.fit_model_fn,
            error_fn=args.error_fn,
            model_kwargs=args.model_kwargs,
            verbose=(not args.quiet))
    else:
        erm_error, over_error, under_error = run_no_projection_model(data_generation_fn=args.data_generation_fn,
            data_args=args.data_args,
            N=args.n_random_features,
            fit_model_fn=args.fit_model_fn,
            error_fn=args.error_fn,
            model_kwargs=args.model_kwargs,
            verbose=(not args.quiet),
            model_file=args.model_file)
       
    save_error_logs(args.outfile,
        [erm_error, over_error, under_error],
        ['ERM', 'oversample', 'undersample'])


def process_args(args):
    # model

    if args.model_type=='ridge':
        fit_model_fn = fit_ridge_regression
        assert args.Lambda
    elif args.model_type=='logistic':
        fit_model_fn = fit_logistic_regression

    model_kwargs = {'Lambda': args.Lambda}
    args.fit_model_fn = fit_model_fn
    args.model_kwargs = model_kwargs

    # data
    if args.toy_example_name=='random_projections':
        data_generation_fn = generate_toy_data_random_projections
        required_args = ['n', 'p_correlation', 'd_causal', 'mean_causal', 'var_causal', 'd_spurious', 'mean_spurious', 'var_spurious']
        random_features = True
    elif args.toy_example_name=='no_projections':
        data_generation_fn = generate_toy_data_no_projections
        required_args = ['n', 'p_correlation', 'mean_causal', 'var_causal', 'mean_spurious', 'var_spurious', 
                         'mean_noise', 'var_noise']
        random_features = False

    assert len(args.n_random_features)>0
    if len(args.n_random_features)==1:
        args.n_random_features = args.n_random_features[0]

    data_args = {}
    for argname in required_args:
        argval = getattr(args, argname)
        assert argval is not None, f'{argname} must be specified'
        data_args[argname] = argval
    args.data_args = data_args
    args.data_generation_fn = data_generation_fn

    # error
    if args.error_type=='zero_one':
        error_fn = zero_one_error
    elif args.error_type=='squared':
        error_fn = squared_error
    args.error_fn = error_fn

    # random features?
    args.random_features = random_features

if __name__=='__main__':
    main()

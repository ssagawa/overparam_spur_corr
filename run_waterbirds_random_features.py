import numpy as np
import argparse
from toy_example_data import *
from random_feature_utils import *

def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--features_path', default=None, required=True)
    parser.add_argument('--metadata_path', default=None, required=True)
    parser.add_argument('-f', '--frac', type=float, default=1.0)
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

    full_data, n_groups = load_waterbirds_data(args.features_path, args.metadata_path, frac=args.frac)
    erm_error, over_error, under_error = run_random_features_model(full_data=full_data,
        n_groups=n_groups,
        N=args.n_random_features,
        fit_model_fn=args.fit_model_fn,
        error_fn=args.error_fn,
        model_kwargs=args.model_kwargs,
        verbose=(not args.quiet))
   
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

    assert len(args.n_random_features)>0
    if len(args.n_random_features)==1:
        args.n_random_features = args.n_random_features[0]

    # error
    if args.error_type=='zero_one':
        error_fn = zero_one_error
    elif args.error_type=='squared':
        error_fn = squared_error
    args.error_fn = error_fn

def load_waterbirds_data(features_path, metadata_path, frac=1.0):
    TRAIN, VAL, TEST = (0, 1, 2)
    features = np.load(features_path)
    metadata = pd.read_csv(metadata_path)
    # Train
    train_mask = metadata['split']==TRAIN
    train_y = metadata[train_mask]['y'].values
    train_x = features[train_mask,:]
    train_g = 2*metadata[train_mask]['y'].values + metadata[train_mask]['place'].values
    if frac < 1:
        idx = np.random.choice(np.arange(train_y.size), int(frac*(train_y.size)))
        train_x = train_x[idx,:]
        train_y = train_y[idx]
        train_g = train_g[idx]
    # Test
    test_mask = metadata['split']==TEST
    test_y = metadata[test_mask]['y'].values
    test_x = features[test_mask,:]
    test_g = 2*metadata[test_mask]['y'].values + metadata[test_mask]['place'].values
    return ((train_x, train_y, train_g), (test_x, test_y, test_g)), 4

if __name__=='__main__':
    main()

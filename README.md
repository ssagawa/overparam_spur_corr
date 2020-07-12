# An Investigation of Why Overparameterization Exacerbates Spurious Correlations 

This code implements the code for the following paper:

> Shiori Sagawa\*, Aditi Raghunathan\*, Pang Wei Koh\*, and Percy Liang
>
> [An Investigation of Why Overparameterization Exacerbates Spurious Correlations](https://arxiv.org/pdf/2005.04345.pdf)

The experiments use the following datasets:
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Waterbirds, formed from [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html) + [Places](http://places2.csail.mit.edu/)

## Abstract
We study why overparameterization---increasing model size well beyond the point of zero training error---can hurt test error on minority groups despite improving average test error when there are spurious correlations in the data.
Through simulations and experiments on two image datasets, we identify two key properties of the training data that drive this behavior: the proportions of majority versus minority groups, and the signal-to-noise ratio of the spurious correlations.
We then analyze a linear setting and theoretically show how the inductive bias of models towards ''memorizing'' fewer examples can cause overparameterization to hurt.
Our analysis leads to a counterintuitive approach of subsampling the majority group, which empirically achieves low minority error in the overparameterized regime, even though the standard approach of upweighting the minority fails.
Overall, our results suggest a tension between using overparameterized models versus using all the training data for achieving low worst-group error.

## Prerequisites
- python 3.6.8
- matplotlib 3.0.3
- numpy 1.16.2
- pandas 0.24.2
- pillow 5.4.1
- pytorch 1.1.0
- pytorch_transformers 1.2.0
- torchvision 0.5.0a0+19315e3
- tqdm 4.32.2

## Datasets and code 

To run our code, you will need to change the `root_dir` variable in `data/data.py`.
Below, we provide sample commands for each dataset.

### CelebA
Our code expects the following files/folders in the `[root_dir]/celebA` directory:

- `data/list_eval_partition.csv`
- `data/list_attr_celeba.csv`
- `data/img_align_celeba/`

You can download these dataset files from [this Kaggle link](https://www.kaggle.com/jessicali9530/celeba-dataset). The original dataset, due to Liu et al. (2015), can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

A sample command to train a model on CelebA is:
`python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width 32`

### Waterbirds

The Waterbirds dataset (Sagawa et al., 2020) is constructed by cropping out birds from photos in the Caltech-UCSD Birds-200-2011 (CUB) dataset (Wah et al., 2011) and transferring them onto backgrounds from the Places dataset (Zhou et al., 2017).

Our code expects the following files/folders in the `[root_dir]/cub` directory:

- `data/waterbird_complete95_forest2water2/`

You can download a tarball of this dataset [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz).

A sample command to train random features model on Waterbirds is:
`python run_waterbirds_random_features.py --features_path resnet18_1layer.npy --metadata_path [root_dir]/cub/data/waterbird_complete95_forest2water2/metadata.csv -N 100 -o results.csv --model_type logistic --error_type zero_one --Lambda 1e-09 --seed 0 --model_file model.pkl`. 

You can extract the features `resnet18_1layer.npy` using `write_pretrained_features.py`.

Note that compared to the training set, the validation and test sets are constructed with different proportions of each group. We describe this in more detail in Appendix C.1 in our [prior paper](https://arxiv.org/abs/1911.08731), which we reproduce here for convenience:

> We use the official train-test split of the CUB dataset, randomly choosing 20% of the training data to serve as a validation set. For the validation and test sets, we allocate distribute landbirds and waterbirds equally to land and water backgrounds (i.e., there are the same number of landbirds on land vs. water backgrounds, and separately, the same number of waterbirds on land vs. water backgrounds). This allows us to more accurately measure the performance of the rare groups, and it is particularly important for the Waterbirds dataset because of its relatively small size; otherwise, the smaller groups (waterbirds on land and landbirds on water) would have too few samples to accurately estimate performance on. We note that we can only do this for the Waterbirds dataset because we control the generation process; for the other datasets, we cannot generate more samples from the rare groups.

> In a typical application, the validation set might be constructed by randomly dividing up the available training data. We emphasize that this is not the case here: the training set is skewed, whereas the validation set is more balanced. We followed this construction so that we could better compare ERM vs. reweighting vs. group DRO techniques using a stable set of hyperparameters. In practice, if the validation set were also skewed, we might expect hyperparameter tuning based on worst-group accuracy to be more challenging and noisy.

> Due to the above procedure, when reporting average test accuracy in our experiments,
we calculate the average test accuracy over each group and then report a weighted average, with weights corresponding to the relative proportion of each group in the (skewed) training dataset.

### Synthetic data with random features (implicit memorization setting)
A sample command for simulations described in Section 4 is:
`python run_toy_example.py -N 100 -o results.csv --toy_example_name random_projections --n 3000 --p_correlation 0.9 --d_causal 100 --mean_causal 1 --var_causal 100 --d_spurious 100 --mean_spurious 1 --var_spurious 1 --model_type logistic --error_type zero_one --Lambda 1e-09 --seed 0`

### Synthetic data with noise features (explicit memorization setting)
A sample command for simulations described in Appendix A.4 is:
`python run_toy_example.py -N 100 -o results.csv --toy_example_name no_projections --n 3000 --p_correlation 0.9 --mean_causal 1 --var_causal 1 --mean_spurious 1 --var_spurious 0.01 --mean_noise 0 --var_noise 1 --model_type logistic --error_type zero_one --Lambda 1e-09 --seed 0 --model_file model.pkl`

# README

## Requirements
- This codebase is written for `python3` and `pytorch`.


## Experiments
### Data
- Please download and place all datasets (CIFAR10, CIFAR100, and TingImageNet) into the data directory. 


### Training

- To train the copula estimator of MI in TinyImageNet (two types of methods)

```
python train_MI_estimator_copulal2.py --model_dir 'your checkpoint directory for MI estimator'
```



- To train the target model using NVC-MIE

```
python train_MIAT_copula.py --model_dir 'your checkpoint directory for the target model'
```


### Test
To test the learned target model

```
python test_comparison.py --model_dir 'your checkpoint directory for the target model'
```



# GCAUV:A Graph Contrastive Learning Framework with Adaptive Augmentation and Encoding for unaligned views
Command for training model on Cora dataset

python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True --drop_scheme=degree

python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True --drop_scheme=pr

python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True --drop_scheme=evc

python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True --drop_scheme=closeness

python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True --drop_scheme=betweenness

![Hyperparameter Settings](https://github.com/GCAUV-PyTorch/GCAUV/blob/main/Hyperparameter%20Settings.jpg)

The image above summarizes the hyperparameter configuration of the model proposed in this paper on all datasets. Where pe,1,pe,2 are
the hyperparameters that control the ratio of edge dropping in the first and second views obtained by sampling; pf,1,pf,2 are the hyperparameter that control
the range of feature masking in the first and second views; τ is the temperature
coefficient; walk lenth is the length of the random walk sampling; and heads is
the number of multi-head attention network heads.

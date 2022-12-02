# GCAUV:A Graph Contrastive Learning Framework with Adaptive Augmentation and Encoding for unaligned views
Command for training model on Cora dataset

python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True --drop_scheme=degree

python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True --drop_scheme=pr

python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True --drop_scheme=evc

python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True --drop_scheme=closeness

python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True --drop_scheme=betweenness


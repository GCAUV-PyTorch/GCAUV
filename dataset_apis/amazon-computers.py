from torch_geometric.datasets import Amazon
import torch_geometric.transforms as T

def load_trainset(trans):
    dataset = Amazon(root='~/datasets', name='computers', transform=T.Compose([trans]))
    return dataset

def load_eval_trainset():
    return Amazon(root='~/datasets', name='computers')

def load_testset():
    return Amazon(root='~/datasets', name='computers')
import random

from torch_geometric.datasets import FacebookPagePage,Planetoid
from torch_geometric.data import Data
import torch
import networkx as nx
import numpy as np
from utils import compute_pr, eigenvector_centrality, closeness_centrality, betweenness_centrality
from torch_geometric.utils import to_undirected, degree, subgraph, remove_isolated_nodes
import pandas as pd



def drop_edge_weighted_closensee(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    # print(edge_weights)
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    # print(edge_weights)
    sel_mask = torch.bernoulli(1.-edge_weights).to(torch.bool)#服从伯努利分布输出0或者1  由原来的1-edge_weights改为edge_weights，因为近性中心性越小越好
    # print(sel_mask)
    return edge_index[:, sel_mask]


def drop_edge_weighted_betweenness(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    # print(edge_weights)
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    # print(edge_weights)
    sel_mask = torch.bernoulli(1.-edge_weights).to(torch.bool)#服从伯努利分布输出0或者1  介中心性越大越好
    # print(sel_mask)
    return edge_index[:, sel_mask]


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    # print(edge_weights)
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    # print(edge_weights)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)#服从伯努利分布输出true或者false
    # print(sel_mask)
    return edge_index[:, sel_mask]


def add_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)  # 服从伯努利分布输出true或者false
    # num = int(edge_index.shape[1] * p)
    num = sel_mask.tolist().count(False)
    drop_edge_index = []
    drop_edge1 = []
    drop_edge2 = []
    for index, keep in enumerate(sel_mask):#获取要删的边的结点索引
        if keep == False:
            drop_edge_index.append(index)

    # print(f"要删除{num}条边")
    #print(drop_edge_index)
    for n in drop_edge_index:
        drop_edge1.append(edge_index[0,n].item())
        drop_edge2.append(edge_index[1,n].item())
    # print(drop_edge1)
    # print(drop_edge2)
    add_edge_point = np.array([drop_edge1, drop_edge2])  # 可以加边的结点
    # print(add_edge_point)


    add_edge1 = []
    add_edge2 = []
    # add_edge1 = random.sample(edge_index[0].tolist(), num)
    # add_edge2 = random.sample(edge_index[1].tolist(), num)
    add_edge1 = random.sample(list(add_edge_point[0]), num)
    add_edge2 = random.sample(list(add_edge_point[1]), num)
    # print(f"添加的边{add_edge}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    add_edge_t = torch.Tensor([add_edge1,add_edge2]).type(torch.int64).to(device)
    # print(add_edge_t)
    edge_index_new = torch.cat((edge_index, add_edge_t), dim=1)#拼接添加边之后的edge_index

    # print(edge_index_new)
    return edge_index_new
def add_edge_weighted_rand(edge_index,edge_weights, p: float, threshold: float = 1.):
    num = int(edge_index.shape[1] * p)
    add_edge1 = []
    add_edge2 = []
    add_edge1 = random.sample(edge_index[0].tolist(), num)
    add_edge2 = random.sample(edge_index[1].tolist(), num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    add_edge_t = torch.Tensor([add_edge1, add_edge2]).type(torch.int64).to(device)
    edge_index_new = torch.cat((edge_index, add_edge_t), dim=1)  # 拼接添加边之后的edge_index
    return edge_index_new
def add_edge_weighted_easy(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)  # 服从伯努利分布输出true或者false
    num = 0
    for keep in sel_mask:
        if keep == False:
            num = num + 1
    add_edge1 = random.sample(edge_index[0].tolist(),num)
    add_edge2 = random.sample(edge_index[1].tolist(),num)
    # print(add_edge1)
    # print(add_edge2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    add_edge_t = torch.Tensor([add_edge1,add_edge2]).type(torch.int64).to(device)
    edge_index_new = torch.cat((edge_index,add_edge_t),dim=1)
    return edge_index_new

def add_edge_weighted_new(edge_index,num):
    add_edge1 = random.sample(edge_index[0].tolist(), num)
    add_edge2 = random.sample(edge_index[1].tolist(), num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    add_edge_t = torch.Tensor([add_edge1, add_edge2]).type(torch.int64).to(device)
    edge_index_new = torch.cat((edge_index, add_edge_t), dim=1)
    return edge_index_new




def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    # print(x.t().size())
    #print(node_c)
    # print(node_c.size())
    if (x.t().size(1) == node_c.size(0)):

        w = x.t() @ node_c  #结点的每个维度与结点中心性相乘 @表示元素与向量的乘法 node_c 乘以 x.t()的转置 1x2708 2708*1433
        #    print(w)
        w = w.log()#归一化
        s = (w.max() - w) / (w.max() - w.mean())#计算每个维度的屏蔽概率
    # print(s)

        return s
    else:
        return torch.ones((x.size(1),))

def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c

    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())
    return s

def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x
def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)
def drop_node_feature_weighted(data, batch ,edge_weights, p1: float, w , p2:float,threshold1: float = 1.,threshold2: float = 0.7):
    #print(f"子图{data.x.size()}")
    edge_weights = edge_weights / edge_weights.mean() * p1
    edge_weights = edge_weights.where(edge_weights < threshold1, torch.ones_like(edge_weights) * threshold1)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)  # 服从伯努利分布输出true或者false

    node_tensor = torch.where(sel_mask == True)[0]
    #print(node_tensor)
    # print(node_tensor.size())
    # print(node_tensor)
    # print(sel_mask.tolist().count(False))
    edge_index_new,_ = subgraph(subset=node_tensor, edge_index=data.edge_index)

    w = w / w.mean() * p2
    w = w.where(w < threshold2, torch.ones_like(w) * threshold2)
    drop_prob = w
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    #num = sel_mask.tolist().count(False)
    #print(num)
    x = data.x.clone()
    x[:, drop_mask] = 0.
    num = 0
    for index, keep in enumerate(sel_mask):  # 获取要删的边的结点索引
            if keep == False:
                 x = del_tensor_ele(x, index - num)
                 batch = del_tensor_ele(batch,index - num)
                 data.original_idx = del_tensor_ele(data.original_idx,index - num)
                 num = num + 1
    np.set_printoptions(threshold=np.inf)
    edge_index_numpy = edge_index_new.cpu().numpy()
    #print(edge_index_numpy)
    add_index = np.zeros((2, edge_index_numpy.shape[1]))
    for index, keep in enumerate(sel_mask):
        if keep == False:
            for k in range(2):
                for i in range(edge_index_numpy.shape[1]):
                    if edge_index_numpy[k][i] >= index:
                        add_index[k][i] = add_index[k][i] + 1

    edge_index_numpy = edge_index_numpy - add_index
    #print(edge_index_numpy)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data.edge_index = torch.from_numpy(edge_index_numpy).type(torch.int64).to(device)
    #print(data.edge_index)
    data.x = x
    return data,batch


def degree_drop_weights_new(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1]).to(torch.float32)#计算每个结点的度
    s_col = torch.log(deg)  # 标准化
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())  # 获得概率
    return weights


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)#在edge_index中添加自环
    deg = degree(edge_index_[1])#计算每个结点的度
    deg_col = deg[edge_index[1]].to(torch.float32)#计算边中心性，当做有向图来计算，边中心性定义为尾结点的结点度数
    s_col = torch.log(deg_col)#标准化
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())#获得概率

    return weights




def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights
def pr_drop_weights_new(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    # pv_row = pv[edge_index[0]].to(torch.float32)
    # pv_col = pv[edge_index[1]].to(torch.float32)
    # s_row = torch.log(pv_row)
    # s_col = torch.log(pv_col)
    s = torch.log(pv)
    # if aggr == 'sink':
    #     s = s_col
    # elif aggr == 'source':
    #     s = s_row
    # elif aggr == 'mean':
    #     s = (s_col + s_row) * 0.5
    # else:
    #     s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights

def evc_drop_weights_new(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    # edge_index = data.edge_index
    # s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    # s = s_col

    return (s.max() - s) / (s.max() - s.mean())
def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())
def closeness_drop_weights(data):
    close = closeness_centrality(data)
    close = close.where(close > 0, torch.zeros_like(close))
    close = close + 1e-8
    s = close.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())
def betweenness_drop_weights(data):
    bet = betweenness_centrality(data)
    bet = bet.where(bet > 0, torch.zeros_like(bet))
    bet = bet + 1e-8
    s = bet.log()


    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col
    return (s.max() - s) / (s.max() - s.mean())

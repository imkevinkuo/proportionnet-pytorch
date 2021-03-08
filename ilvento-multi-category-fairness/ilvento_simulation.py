import torch
import math


def proportional_allocation(x, l):
    # x is a L x N x M matrix. x[l][i][j] = lth sample of i's bid on j
    # z: L x M. z[i][j] = total allocation score of item j, ith sample
    L, N, M = x.shape
    xl = x**l
    z = xl.sum(dim=1).unsqueeze(dim=1)
    alloc = xl / z
    return alloc


# The value of l satisfying total variation fairness on uniform distance d.
def l_upper_bound(bid_ratio, d):
    return (math.log(1+d) - math.log(1-d)) / math.log(bid_ratio)


def generate_dataset_nxk(n_agents, n_items, num_examples, item_ranges):
    range_diff = item_ranges[:, :, 1] - item_ranges[:, :, 0]
    return range_diff * torch.rand(num_examples, n_agents, n_items) + item_ranges[:, :, 0]


# Assume second price auction?
def get_revenue(batch, alloc):
    sec_price = batch.min(dim=1).values.unsqueeze(dim=1) # L x 1 x M
    revenue = sec_price*alloc
    return revenue


def get_revenue_vval(batch, alloc):
    vval = 2*batch - torch.Tensor([[2,3],[3,2]])
    return (vval*alloc)


def get_welfare(batch, alloc):
    welfare = batch*alloc
    return welfare


def get_revenue_vcg(batch, alloc):
    welfare = batch*alloc
    # Bidder 1 does not participate - bidder 2 always wins, allocation is 1
    welfare1 = batch[:, [1, 1], :]*torch.Tensor([[[0], [1]]])
    welfare2 = batch[:, [0, 0], :]*torch.Tensor([[[1], [0]]])
    #print(batch[0])
    #print(welfare1[0] - welfare[0])
    #print(welfare2[0] - welfare[0])
    #print()
    return welfare1 + welfare2 - 2*welfare


batch = generate_dataset_nxk(2, 2, 640000, torch.Tensor([[[1,2], [2,3]], [[2,3], [1,2]]]))
for d in [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9999999]:
    l = l_upper_bound(3, d)
    alloc = proportional_allocation(batch, l)
    welfare = get_welfare(batch, alloc).mean(dim=0)
    revenue = get_revenue(batch, alloc).mean(dim=0)
    print("d:", d, l)
    print("Allocs:", alloc.mean(dim=0))
    print("Welfare:", welfare, welfare.sum())
    print("Revenue:", revenue, revenue.sum())
    print()

import torch
from torch import nn
from regretnet.utils import calc_agent_util
from fairness import featuresets


""" Total variation fairness """


def setup_fairness(args, device):
    if args.fairness:
        if args.fairness[0] == 'tvf':
            if len(args.fairness) == 3:
                if args.fairness[1] == 'fr1':
                    C, D = featuresets.experiment_1(float(args.fairness[2]))
                elif args.fairness[1] == 'fr2':
                    C, D = featuresets.experiment_2(float(args.fairness[2]))
                elif args.fairness[1] == 'fr3':
                    C, D = featuresets.experiment_3(float(args.fairness[2]))
                else:
                    _, c_name, d_name = args.fairness
                    afeats = torch.load(c_name + '.pt').astype(int)  # file for ad prefs
                    ufeats = torch.load(d_name + '.pt').astype(int)  # file for user qualities
                    C = featuresets.load_categories(afeats)
                    D = featuresets.generate_distance(afeats, ufeats).to(device)
            if len(args.fairness) == 2:
                # Single category, uniform distance
                _, d = args.fairness
                C = featuresets.single_category(args.n_agents)
                D = featuresets.uniform_distance(1, args.n_items, float(d)).to(device)
            # print(D)
            return ['tvf', C, D]
        if args.fairness[0] == 'cvg':
            if len(args.fairness) == 2:
                _, c = args.fairness
                return ['cvg', float(c)]
    return []


def get_unfairness(batch, allocs, payments, fairness, factor=1, coverage_allocs=None):
    # factor is from 0 - 1, for gradual application as epochs increase

    # Later on we can specify names for these different types of loss functions
    # if fairness_type == 'maximin':
    #     return fairness_maximin(batch, allocs, payments, *fairness_params)
    # if fairness_type == 'alloc_restrict':
    #     return fairness_alloc_restrict(batch, allocs, payments, *fairness_params)
    # if fairness_type == 'competitive_bid':
    #     return fairness_competitive_bid(batch, allocs, payments, *fairness_params)
    # if fairness_type == 'exorbitant_bid':
    #     return fairness_exorbitant_bid(batch, allocs, payments, *fairness_params)
    # if fairness_type == 'bid_proportional':
    #     return fairness_bid_proportional(batch, allocs, payments, *fairness_params)
    # if fairness_type == 'alloc_competitive_restrict':
    #     return fairness_alloc_competitive_restrict(batch, allocs, payments, *fairness_params)
    if fairness:
        if fairness[0] == 'total_variation' or fairness[0] == 'tvf':
            unfairness = fairness_distance(batch, allocs, payments, fairness[1], fairness[2], factor)
        elif fairness[0] == 'coverage' or fairness[0] == 'cvg':
            unfairness = fairness_coverage(coverage_allocs, fairness[1])
        else:
            unfairness = torch.zeros(batch.shape[0], batch.shape[2]).to(allocs.device)
    else:
        unfairness = torch.zeros(batch.shape[0], batch.shape[1])

    # assert unfairness.shape == (batch.shape[0], batch.shape[2])
    return unfairness


def fairness_distance(batch, allocs, payments, C, D, factor):
    L, n, m = allocs.shape
    unfairness = torch.zeros(L, m).to(allocs.device)
    for u in range(m):
        for v in range(m):
            for i, C_i in enumerate(C):
                # ReLU of difference
                subset_allocs_diff = (allocs[:, C_i, v] - allocs[:, C_i, u]).clamp_min(0)
                # If allocation distance is greater than user distance, penalize
                D2 = 1 - (1 - D) * factor
                unfairness[:, u] += (subset_allocs_diff.sum(dim=1) - D2[i, u, v]).clamp_min(0)
    return unfairness


# Get the max L1 norm variation within each category - useful to track when using uniform distance
def max_variation(batch, allocs, payments, fairness_args):
    L, n, m = allocs.shape
    C = fairness_args[1] if fairness_args[0] == 'tvf' and len(fairness_args) > 1 else featuresets.single_category(n)
    variation = torch.zeros(L, len(C)).to(allocs.device)
    for u in range(m):
        for v in range(m):
            for i, C_i in enumerate(C):
                subset_allocs_diff = (allocs[:, C_i, u] - allocs[:, C_i, v]).abs().view(L, len(C_i)).sum(dim=1)
                variation[:, i] = variation[:, i].max(subset_allocs_diff)
    return variation


def fairness_coverage(allocs, c):
    # TODO: C: NxMx2. C[i][j] = (lower, upper) bound on coverage proportion for agent i over item type j.
    # For now c: [0, 0.5] is just uniform to all entries in C.
    coverage = allocs.sum(dim=0)  # NxM, coverage[i][j] = coverage of agent i over item type j.
    n, m = coverage.shape
    total_coverage = coverage.sum(dim=1).unsqueeze(dim=1)  # N, total_coverage[i] = expected number of items won by agent i.
    ratio_coverage = coverage / total_coverage  # NxM: proportional coverage
    lower_violation = (c - ratio_coverage).clamp_min(0)
    upper_violation = (ratio_coverage - (1-c)).clamp_min(0)
    total_violation = lower_violation + upper_violation
    return total_violation.sum(dim=0).reshape(1, m)  # Get an unfairness value for each item


""" Unused functions """


def fairness_maximin(batch, allocs, payments, d=0.5):
    """ Maximizes the minimum utility in order to satisfy d.
    d: maximum allowed difference of utility between any two agents. """
    agent_utils = calc_agent_util(batch, allocs, payments)
    max_agent_utils = agent_utils.max(dim=1).values
    min_agent_utils = agent_utils.min(dim=1).values
    return (-d + max_agent_utils - min_agent_utils).clamp_min(min=0)


def fairness_alloc_restrict(batch, allocs, payments, c=0.7):
    """ c: maximum allowed allocation probability for any agent. """
    return (-c + allocs.sum(dim=2)).clamp_min(min=0).sum(dim=1)


def fairness_competitive_bid(batch, allocs, payments, c=0.7, d=0.5):
    """ Maximin with a required "competitive" bid threshold.
    c: ratio of highest bid to be considered competitive
    d: maximum allocation difference between any competitive bid vs. max allocation."""
    # batch shape: (L samples, N agents, M items)
    # samples x items, each element is c*max bid
    cutoff_bid_item = c * batch.max(dim=1, keepdim=True).values
    # competitiveness below cutoff bid = 0, at max bid = 1.
    competitiveness = ((batch - cutoff_bid_item) / (1 - cutoff_bid_item)).clamp_min(min=0)
    # allocations shape: (n_agents (+1 dummy), M items)
    allocation_disp = (-d + allocs.max(dim=1, keepdim=True).values - allocs).clamp_min(min=0)
    return (competitiveness * allocation_disp).sum(dim=(1, 2))


def fairness_exorbitant_bid(batch, allocs, payments, d=0.5):
    """ Maximin with a required "exorbitant" bid threshold.
    c: ratio of highest bid to be considered competitive
    d: maximum allocation difference between any exorbitant bid vs. max allocation. """
    bid_proportions = batch / batch.sum(dim=2, keepdim=True)
    allocation_disp = (-d + allocs.max(dim=1, keepdim=True).values - allocs).clamp_min(min=0)
    return (bid_proportions * allocation_disp).sum(dim=(1, 2))


def fairness_bid_proportional(batch, allocs, payments, c=0.7):
    """ Bidder's allocation must be proportional to their share of bids.
    c: e.g. our bids take up 40% of auction share, but we only receive 20% of allocations.
    Setting c > 0.5 penalizes the network, but c < 0.5 does not."""
    alloc_proportion = allocs.sum(dim=2, keepdim=True) / allocs.shape[2]
    bid_proportion = batch.sum(dim=2, keepdim=True) / batch.sum(dim=(1,2), keepdim=True)
    return ((c * bid_proportion) - alloc_proportion).clamp_min(min=0).sum(dim=1)


def fairness_alloc_competitive_restrict(batch, allocs, payments, c=0.7, L=.55):
    """ Because of sigmoid think of L > 0.5 as one allocated item """
    m = nn.Sigmoid()
    unfair_allocs = m(allocs - (allocs.max(1, True)[0] * c))
    return (-L + unfair_allocs.sum(2)).clamp_min(0).sum(1)

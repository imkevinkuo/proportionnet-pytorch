import torch


class Dataloader(object):
    def __init__(self, data, batch_size=64, shuffle=True):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.size = data.size(0)
        self.data = data
        self.iter = 0

    def _sampler(self, size, batch_size, shuffle=True):
        if shuffle:
            idxs = torch.randperm(size)
        else:
            idxs = torch.arange(size)
        for batch_idxs in idxs.split(batch_size):
            yield batch_idxs

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter == 0:
            self.sampler = self._sampler(self.size, self.batch_size, shuffle=self.shuffle)
        self.iter = (self.iter + 1) % (len(self) + 1)
        idx = next(self.sampler)
        return self.data[idx]

    def __len__(self):
        return (self.size - 1) // self.batch_size + 1


def dataset_override(args):
    # Preset multiple variables with dataset name
    if args.dataset:
        if args.dataset[0].startswith('1x2'):
            args.n_agents = 1
            args.n_items = 2
            if 'pv' in args.dataset[0]:
                args.unit = True
        if args.dataset[0].startswith("2x2"):
            args.n_agents = 2
            args.n_items = 2
        if not args.fairness:
            if args.n_agents == 1:
                args.fairness = ['tvf', '1']
            else:
                args.fairness = ['tvf', '2']
        if args.name == 'testing_name':
            args.name = '_'.join([str(x) for x in args.dataset] +
                                 [str(x) for x in args.fairness[1:]] +
                                 [str(args.random_seed)])


def preset_valuation_range(n_agents, n_items, dataset=None):
    # defaults
    zeros = torch.zeros(n_agents, n_items)
    ones = torch.ones(n_agents, n_items)
    item_ranges = torch.stack((zeros, ones), dim=2).reshape(n_agents, n_items, 2)
    # modifications
    if dataset:
        if 'manelli' in dataset[0] or 'mv' in dataset[0]:
            pass
            # offset = float(dataset[1]) if len(dataset) > 1 else 1
            # item_ranges[:, :, 1] = item_ranges[:, :, 1] + offset
        elif 'pavlov' in dataset[0] or 'pv' in dataset[0]:
            offset = float(dataset[1]) if len(dataset) > 1 else 0
            item_ranges = item_ranges + offset
        elif dataset[0] == 'fr':
            offset = float(dataset[1]) if len(dataset) > 1 else 0
            item_ranges[:, 2, :] = item_ranges[:, 2, :] + offset
            item_ranges[:, 3, :] = item_ranges[:, 3, :] + offset
        elif dataset[0] == '2x2-opp':
            item_ranges[0, 0, :] = item_ranges[0, 0, :] + 1
            item_ranges[0, 1, :] = item_ranges[0, 1, :] + 2
            item_ranges[1, 0, :] = item_ranges[1, 0, :] + 2
            item_ranges[1, 1, :] = item_ranges[1, 1, :] + 1
        # elif dataset[0] == 'frc':
        #     offset = float(dataset[1]) if len(dataset) > 1 else 0
        #     item_ranges[0, 2, :] = item_ranges[0, 2, :] + offset
        #     item_ranges[0, 3, :] = item_ranges[0, 3, :] + offset
        #     #
        #     item_ranges[1, 2, :] = item_ranges[1, 2, :] + offset
        #     item_ranges[1, 3, :] = item_ranges[1, 3, :] + offset
        #     # #
        #     item_ranges[2, 0, :] = item_ranges[2, 0, :] + offset
        #     item_ranges[2, 2, :] = item_ranges[2, 2, :] + offset
        else:
            print(dataset[0], 'is not a valid dataset name. Defaulting to Manelli-Vincent auction.')
    # item_ranges is a n_agents x n_items x 2 tensor where item_ranges[agent_i][item_j] = [lower_bound, upper_bound].
    assert item_ranges.shape == (n_agents, n_items, 2)
    return item_ranges


def generate_linspace_nxk(n_agents, n_items, item_ranges, s=100):
    # For 2-item auctions only.
    b1 = torch.linspace(*item_ranges[0, 0], s)
    b2 = torch.linspace(*item_ranges[0, 1], s)
    return torch.stack(torch.meshgrid([b1, b2]), dim=2).reshape(s**2, 1, 2)


def generate_dataset_nxk(n_agents, n_items, num_examples, item_ranges):
    range_diff = item_ranges[:, :, 1] - item_ranges[:, :, 0]
    return range_diff * torch.rand(num_examples, n_agents, n_items) + item_ranges[:, :, 0]


def get_clamp_op(item_ranges: torch.Tensor):
    def clamp_op(batch):
        samples, n_agents, n_items = batch.shape
        for i in range(n_agents):
            for j in range(n_items):
                lower = item_ranges[i, j, 0]
                upper = item_ranges[i, j, 1]
                batch[:, i, j] = batch[:, i, j].clamp_min(lower).clamp_max(upper)
    return clamp_op

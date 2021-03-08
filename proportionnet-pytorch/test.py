import os
from argparse import ArgumentParser
import torch
import numpy as np
from regretnet.datasets import generate_dataset_nxk, preset_valuation_range, generate_linspace_nxk, get_clamp_op
from regretnet.regretnet import RegretNet, test_loop, RegretNetUnitDemand
from regretnet.datasets import Dataloader
import json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--test-num-examples', type=int, default=3000)
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--test-batch-size', type=int, default=512)
parser.add_argument('--misreport-lr', type=float, default=2e-2)
parser.add_argument('--misreport-iter', type=int, default=25)
parser.add_argument('--test-misreport-iter', type=int, default=1000)
parser.add_argument('--name', default="")
parser.add_argument('--n-agents', default=1)
parser.add_argument('--n-items', default=2)
# Fairness
parser.add_argument('--fairness', nargs='+', default=[])  # Fairness metric and associated parameters
# Plotting
parser.add_argument('--save-all', action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    checkpoint = torch.load(args.name)
    train_args = checkpoint['args']
    arch = checkpoint['arch']

    print("Architecture:")
    arch['activation'] = 'relu'
    print(json.dumps(arch, indent=4, sort_keys=True))
    print("Training Args:")
    print(json.dumps(vars(train_args), indent=4, sort_keys=True))

    args.dataset = train_args.dataset
    if not args.fairness:
        args.fairness = train_args.fairness

    item_ranges = preset_valuation_range(arch['n_agents'], arch['n_items'], train_args.dataset)
    clamp_op = get_clamp_op(item_ranges)
    if train_args.unit:
        model = RegretNetUnitDemand(arch, clamp_op=clamp_op).to(DEVICE)
    else:
        model = RegretNet(arch, clamp_op=clamp_op).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if arch['n_agents'] == 1 and arch['n_items'] == 2:
        test_data = generate_linspace_nxk(arch['n_agents'], arch['n_items'], item_ranges)
    else:
        test_data = generate_dataset_nxk(arch['n_agents'], arch['n_items'], args.test_num_examples, item_ranges).to(DEVICE)

    test_loader = Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

    result = test_loop(model, test_loader, args, device=DEVICE, coverage_data=test_data)
    print(f"Experiment:{checkpoint['name']}")
    print(json.dumps(result, indent=4, sort_keys=True))

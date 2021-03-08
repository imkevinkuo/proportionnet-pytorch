import os
from argparse import ArgumentParser
import torch
import numpy as np

from regretnet import datasets as ds
from regretnet.regretnet import RegretNet, train_loop, test_loop, RegretNetUnitDemand
from torch.utils.tensorboard import SummaryWriter
from regretnet.datasets import Dataloader
import json
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--num-examples', type=int, default=160000)
parser.add_argument('--test-num-examples', type=int, default=10000)
parser.add_argument('--test-iter', type=int, default=5)
parser.add_argument('--n-agents', type=int, default=1)
parser.add_argument('--n-items', type=int, default=2)
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=128)
parser.add_argument('--model-lr', type=float, default=1e-3)
parser.add_argument('--misreport-lr', type=float, default=1e-1)
parser.add_argument('--misreport-iter', type=int, default=25)
parser.add_argument('--test-misreport-iter', type=int, default=1000)
parser.add_argument('--payment_power', type=float, default=0.)
parser.add_argument('--rho', type=float, default=1.0)
parser.add_argument('--rho-incr-iter', type=int, default=5000)
parser.add_argument('--rho-incr-amount', type=float, default=1.0)
parser.add_argument('--lagr-update-iter', type=int, default=100)
# Fairness
parser.add_argument('--fairness', default=[], nargs='+')  # Fairness metric and associated arguments
parser.add_argument('--fair-full', type=int, default=0)  # Gradual enforcement of fairness
parser.add_argument('--rho-fair', type=float, default=1.0)
parser.add_argument('--rho-incr-iter-fair', type=int, default=5000)
parser.add_argument('--rho-incr-amount-fair', type=float, default=1.0)
parser.add_argument('--lagr-update-iter-fair', type=int, default=100)
# Dataset: specifies a valuation config
parser.add_argument('--dataset', nargs='+', default=[])
parser.add_argument('--resume', default="")
parser.add_argument('--resume-epoch', type=int, default=0)
# architectural arguments
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--hidden-layer-size', type=int, default=100)
parser.add_argument('--n-hidden-layers', type=int, default=2)
parser.add_argument('--separate', action='store_true')
parser.add_argument('--name', default='testing_name')
parser.add_argument('--unit', action='store_true')  # saved in model['args'], not in arch
# Plotting
parser.add_argument('--save-all', action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    print(device)

    if args.fair_full == -1:
        args.fair_full = args.num_epochs + 1

    # Replaces n_items, n_agents, name
    ds.dataset_override(args)

    # Valuation range setup
    item_ranges = ds.preset_valuation_range(args.n_agents, args.n_items, args.dataset)
    clamp_op = ds.get_clamp_op(item_ranges)

    # TODO: Need to carry over loaded arch values
    if args.unit:
        model = RegretNetUnitDemand(args.__dict__, clamp_op, device).to(device)
    else:
        model = RegretNet(args.__dict__, clamp_op, device).to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.load_mults(checkpoint['mults'])

    if not os.path.exists("result"):
        os.mkdir("result")

    # If name already exists, append a number to it
    orig_name, i = args.name, 1
    while os.path.exists(f'result/{args.name}'):
        args.name = f'{orig_name}_{i}'
        i += 1
    os.mkdir(f"result/{args.name}")
    writer = SummaryWriter(log_dir=f"run/{args.name}", comment=f"{args}")

    train_data = ds.generate_dataset_nxk(args.n_agents, args.n_items, args.num_examples, item_ranges).to(device)
    train_loader = Dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = ds.generate_dataset_nxk(args.n_agents, args.n_items, args.test_num_examples, item_ranges).to(device)
    test_loader = Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

    print("Training Args:")
    print(json.dumps(vars(args), indent=4, sort_keys=True))
    train_loop(model, train_loader, test_loader, args, writer, device=device, coverage_data=train_data[:40000])
    writer.close()

    result = test_loop(model, test_loader, args, device=device)
    print(f"Experiment:{args.name}")
    print(json.dumps(result, indent=4, sort_keys=True))

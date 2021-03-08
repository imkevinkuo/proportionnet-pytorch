import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm as tqdm

from regretnet.utils import optimize_misreports, tiled_misreport_util, calc_agent_util
from fairness import fairness
import torch.nn.init


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class View_Cut(nn.Module):
    def __init__(self):
        super(View_Cut, self).__init__()

    def forward(self, x):
        return x[:, :-1, :]


class RegretNetUnitDemand(nn.Module):
    def __init__(self, args, clamp_op, device='cpu'):
        super(RegretNetUnitDemand, self).__init__()
        if args['activation'] == 'tanh':
            self.act = nn.Tanh
            self.act_name = 'tanh'
        else:
            self.act = nn.ReLU
            self.act_name = 'relu'

        self.clamp_op = clamp_op

        self.n_agents = args['n_agents']
        self.n_items = args['n_items']

        self.input_size = self.n_agents * self.n_items
        self.hidden_layer_size = args['hidden_layer_size']
        self.n_hidden_layers = args['n_hidden_layers']
        self.separate = args['separate']

        # outputs are agents (+dummy agent) per item (+ dummy item), plus payments per agent
        self.allocations_size = (self.n_agents + 1) * (self.n_items + 1)
        self.payments_size = self.n_agents

        self.nn_model = nn.Sequential(
            *([nn.Linear(self.input_size, self.hidden_layer_size), self.act()] +
              [l for i in range(self.n_hidden_layers)
               for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())])
        )

        self.allocation_head = nn.Linear(self.hidden_layer_size, self.allocations_size * 2)
        self.fractional_payment_head = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.payments_size), nn.Sigmoid()
        )

        # Hyperparams
        self.payment_mult = 1
        self.regret_mults = 5.0 * torch.ones((1, self.n_agents)).to(device)
        self.fair_mults = torch.ones((1, self.n_items)).to(device)
        if 'rho' in args:
            self.rho = args['rho']
        if 'rho_fair' in args:
            self.rho_fair = args['rho_fair']

    def glorot_init(self):
        """
        reinitializes with glorot (aka xavier) uniform initialization
        """

        def initialize_fn(layer):
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)

        self.apply(initialize_fn)

    def forward(self, reports):
        x = reports.view(-1, self.n_agents * self.n_items)
        x = self.nn_model(x)

        alloc_scores = self.allocation_head(x)
        alloc_first = F.softmax(alloc_scores[:, 0:self.allocations_size].view(-1, self.n_agents + 1, self.n_items + 1),
                                dim=1)
        alloc_second = F.softmax(
            alloc_scores[:, self.allocations_size:self.allocations_size * 2].view(-1, self.n_agents + 1,
                                                                                  self.n_items + 1), dim=2)
        allocs = torch.min(alloc_first, alloc_second)

        payments = self.fractional_payment_head(x) * torch.sum(
            allocs[:, :-1, :-1] * reports, dim=2
        )

        return allocs[:, :-1, :-1], payments

    def arch(self):
        return {
            'n_agents': self.n_agents,
            'n_items': self.n_items,
            'hidden_layer_size': self.hidden_layer_size,
            'n_hidden_layers': self.n_hidden_layers,
            'activation': self.act_name,
            'separate': self.separate
        }

    def mults(self):
        return {
            'payment_mult': self.payment_mult,
            'regret_mults': self.regret_mults,
            'fair_mults': self.fair_mults,
            'rho': self.rho,
            'rho_fair': self.rho_fair
        }

    def load_mults(self, mults):
        self.payment_mult = mults['payment_mult']
        self.regret_mults = mults['regret_mults']
        self.fair_mults = mults['fair_mults']
        self.rho = mults['rho']
        # self.rho_fair = mults['rho_fair']
        self.rho_fair = 1


class RegretNet(nn.Module):
    # args is a dict so we can test without argparse
    def __init__(self, args, clamp_op=None, device='cpu'):
        super(RegretNet, self).__init__()

        # this is for additive valuations only
        if args['activation'] == 'tanh':
            self.act = nn.Tanh
            self.act_name = 'tanh'
        else:
            self.act = nn.ReLU
            self.act_name = 'relu'

        self.clamp_op = clamp_op

        self.n_agents = args['n_agents']
        self.n_items = args['n_items']

        self.input_size = self.n_agents * self.n_items
        self.hidden_layer_size = args['hidden_layer_size']
        self.n_hidden_layers = args['n_hidden_layers']
        self.separate = args['separate']

        # outputs are agents (+dummy agent) per item, plus payments per agent
        self.allocations_size = (self.n_agents + 1) * self.n_items
        self.payments_size = self.n_agents

        # Set a_activation to softmax
        self.allocation_head = [nn.Linear(self.hidden_layer_size, self.allocations_size),
                                View((-1, self.n_agents + 1, self.n_items)),
                                nn.Softmax(dim=1),
                                View_Cut()]

        # Set p_activation to frac_sigmoid
        self.payment_head = [
            nn.Linear(self.hidden_layer_size, self.payments_size), nn.Sigmoid()
        ]

        if self.separate:
            self.nn_model = nn.Sequential()
            self.payment_head = [nn.Linear(self.input_size, self.hidden_layer_size), self.act()] + \
                                [l for i in range(self.n_hidden_layers)
                                 for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())] + \
                                self.payment_head

            self.payment_head = nn.Sequential(*self.payment_head)
            self.allocation_head = [nn.Linear(self.input_size, self.hidden_layer_size), self.act()] + \
                                   [l for i in range(self.n_hidden_layers)
                                    for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())] + \
                                   self.allocation_head
            self.allocation_head = nn.Sequential(*self.allocation_head)
        else:
            self.nn_model = nn.Sequential(
                *([nn.Linear(self.input_size, self.hidden_layer_size), self.act()] +
                  [l for i in range(self.n_hidden_layers)
                   for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())])
            )
            self.allocation_head = nn.Sequential(*self.allocation_head)
            self.payment_head = nn.Sequential(*self.payment_head)

        # Hyperparams
        self.payment_mult = 1
        self.regret_mults = 5.0 * torch.ones((1, self.n_agents)).to(device)
        self.fair_mults = torch.ones((1, self.n_items)).to(device)
        if 'rho' in args:
            self.rho = args['rho']
        if 'rho_fair' in args:
            self.rho_fair = args['rho_fair']

    def glorot_init(self):
        """
        reinitializes with glorot (aka xavier) uniform initialization
        """

        def initialize_fn(layer):
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)

        self.apply(initialize_fn)

    def forward(self, reports):
        # x should be of size [batch_size, n_agents, n_items
        # should be reshaped to [batch_size, n_agents * n_items]
        # output should be of size [batch_size, n_agents, n_items],
        # either softmaxed per item, or else doubly stochastic
        x = reports.view(-1, self.n_agents * self.n_items)
        x = self.nn_model(x)
        allocs = self.allocation_head(x)

        # frac_sigmoid payment: multiply p = p_tilde * sum(alloc*bid)
        payments = self.payment_head(x) * torch.sum(
            allocs * reports, dim=2
        )

        return allocs, payments

    def arch(self):
        return {
            'n_agents': self.n_agents,
            'n_items': self.n_items,
            'hidden_layer_size': self.hidden_layer_size,
            'n_hidden_layers': self.n_hidden_layers,
            'activation': self.act_name,
            'separate': self.separate
        }

    def mults(self):
        return {
            'payment_mult': self.payment_mult,
            'regret_mults': self.regret_mults,
            'fair_mults': self.fair_mults,
            'rho': self.rho,
            'rho_fair': self.rho_fair
        }

    def load_mults(self, mults):
        self.payment_mult = mults['payment_mult']
        self.regret_mults = mults['regret_mults']
        self.fair_mults = mults['fair_mults']
        self.rho = mults['rho']
        self.rho_fair = 1
        # self.rho_fair = mults['rho_fair']


def test_loop(model, loader, args, device='cpu', coverage_data=None):
    fairness_args = fairness.setup_fairness(args, device)
    # regrets, payments (n_samples, n_agents)
    # unfairs (n_samples, )
    test_regrets = torch.Tensor().to(device)
    test_payments = torch.Tensor().to(device)
    test_unfairs = torch.Tensor().to(device)
    test_variations = torch.Tensor().to(device)
    test_batch = torch.Tensor().to(device)
    test_allocs = torch.Tensor().to(device)
    test_welfares = torch.Tensor().to(device)

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        misreport_batch = optimize_misreports(model, batch, misreport_iter=args.test_misreport_iter, lr=args.misreport_lr)

        allocs, payments = model(batch)
        welfare = batch*allocs
        truthful_util = calc_agent_util(batch, allocs, payments)
        misreport_util = tiled_misreport_util(misreport_batch, batch, model)

        regrets = misreport_util - truthful_util
        positive_regrets = torch.clamp_min(regrets, 0)
        # unfairs = fairness.get_unfairness(batch, allocs, payments, fairness_args, coverage_allocs=coverage_data)
        variations = fairness.max_variation(batch, allocs, payments, fairness_args)

        # Record entire test data
        test_regrets = torch.cat((test_regrets, positive_regrets), dim=0)
        test_payments = torch.cat((test_payments, payments), dim=0)
        # test_unfairs = torch.cat((test_unfairs, unfairs), dim=0)
        test_variations = torch.cat((test_variations, variations), dim=0)
        test_welfares = torch.cat((test_welfares, welfare), dim=0)
        test_allocs = torch.cat((test_allocs, allocs), dim=0)

        if args.save_all:
            test_batch = torch.cat((test_batch, batch), dim=0)

    test_unfairs = fairness.get_unfairness(None, None, None, fairness_args, coverage_allocs=test_allocs)
    result = {
        "payment_mean": test_payments.sum(dim=1).mean(dim=0).item(),
        "payment_std": test_payments.sum(dim=1).std().item(),
        "regret_mean": test_regrets.sum(dim=1).mean(dim=0).item(),
        "regret_std": test_regrets.sum(dim=1).std().item(),
        "regret_max": test_regrets.sum(dim=1).max().item(),
        "unfairness_mean": test_unfairs.sum(dim=1).mean().item(),
        "unfairness_max": test_unfairs.sum(dim=1).max().item(),
        "unfairness_std": test_unfairs.sum(dim=1).std().item(),
        "variation_max": test_variations.max().item(),
        "welfare": test_welfares.mean(dim=0).sum().item()
    }

    if args.save_all:
        torch.save(test_batch, f"{args.name}_batch.pt")
        torch.save(test_allocs, f"{args.name}_alloc.pt")
        torch.save(test_unfairs, f"{args.name}_unfair.pt")
        torch.save(test_payments, f"{args.name}_payment.pt")
        torch.save(test_regrets, f"{args.name}_regret.pt")
    # for i in range(model.n_agents):
    #     agent_regrets = test_regrets[:, i]
    #     result[f"regret_agt{i}_std"] = (((agent_regrets ** 2).mean() - agent_regrets.mean() ** 2) ** .5).item()
    #     result[f"regret_agt{i}_mean"] = agent_regrets.mean().item()
    return result


def train_loop(model, train_loader, test_loader, args, writer, device="cpu", coverage_data=None):
    fairness_args = fairness.setup_fairness(args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)
    print(fairness_args)
    iter = 0
    for epoch in tqdm(range(args.num_epochs)):
        regrets_epoch = torch.Tensor().to(device)
        payments_epoch = torch.Tensor().to(device)
        unfairness_epoch = torch.Tensor().to(device)
        variation_epoch = torch.Tensor().to(device)
        welfare_epoch = torch.Tensor().to(device)
        # Calculate coverage...?
        allocs_epoch = torch.Tensor().to(device)
        for i, batch in enumerate(train_loader):
            iter += 1
            batch = batch.to(device)
            misreport_batch = optimize_misreports(model, batch, misreport_iter=args.misreport_iter, lr=args.misreport_lr)

            allocs, payments = model(batch)
            welfare = batch*allocs
            truthful_util = calc_agent_util(batch, allocs, payments)
            misreport_util = tiled_misreport_util(misreport_batch, batch, model)
            regrets = misreport_util - truthful_util
            positive_regrets = torch.clamp_min(regrets, 0)

            payment_loss = payments.sum(dim=1).mean() * model.payment_mult
            variations = fairness.max_variation(batch, allocs, payments, fairness_args)

            coverage_allocs, _ = model(coverage_data)
            unfairness = fairness.get_unfairness(batch, allocs, payments, fairness_args,
                                                 min(1, epoch/args.fair_full) if args.fair_full > 0 else 1,
                                                 coverage_allocs)

            regret_loss = (model.regret_mults * positive_regrets).mean()
            regret_quad = (model.rho / 2.0) * (positive_regrets ** 2).mean()
            # Also try to push down the max_regret
            # regret_loss = (regret_mults * (positive_regrets + positive_regrets.max(dim=0).values) / 2).mean()
            # regret_quad = (rho / 2.0) * ((positive_regrets ** 2).mean() +
            #                              (positive_regrets.max(dim=0).values ** 2).mean()) / 2

            unfairness_loss = (model.fair_mults * unfairness).mean()
            unfairness_quad = (model.rho_fair / 2.0) * (unfairness ** 2).mean()

            # Add batch to epoch stats
            regrets_epoch = torch.cat((regrets_epoch, regrets), dim=0)
            payments_epoch = torch.cat((payments_epoch, payments), dim=0)
            variation_epoch = torch.cat((variation_epoch, variations), dim=0)
            unfairness_epoch = torch.cat((unfairness_epoch, unfairness), dim=0)
            welfare_epoch = torch.cat((welfare_epoch, welfare), dim=0)
            allocs_epoch = torch.cat((allocs_epoch, allocs), dim=0)

            # Calculate loss
            loss_func = regret_loss \
                        + regret_quad \
                        - payment_loss \
                        + unfairness_loss \
                        + unfairness_quad

            # update model
            optimizer.zero_grad()
            loss_func.backward()
            optimizer.step()

            # update various fancy multipliers
            if iter % args.lagr_update_iter == 0:
                with torch.no_grad():
                    model.regret_mults += model.rho * positive_regrets.mean(dim=0)
            if iter % args.rho_incr_iter == 0:
                model.rho += args.rho_incr_amount

            if iter % args.lagr_update_iter_fair == 0:
                with torch.no_grad():
                    model.fair_mults += model.rho_fair * unfairness.mean(dim=0)
            if iter % args.rho_incr_iter_fair == 0:
                model.rho_fair += args.rho_incr_amount_fair

        # Log testing stats and save model
        if epoch % args.test_iter == (args.test_iter - 1):
            test_result = test_loop(model, test_loader, args, device=device)
            for key, value in test_result.items():
                writer.add_scalar(f"test/{key}", value, global_step=epoch + args.resume_epoch)

        torch.save({'name': args.name,
                    'arch': model.arch(),
                    'mults': model.mults(),
                    'state_dict': model.state_dict(),
                    'args': args},
                   f"result/{args.name}/{epoch + args.resume_epoch}_checkpoint.pt")

        # Log training stats
        train_stats = {
            "regret_max": regrets_epoch.max().item(),
            "regret_mean": regrets_epoch.sum(dim=1).mean().item(),
            "regret_std": regrets_epoch.sum(dim=1).std().item(),

            "payment": payments_epoch.sum(dim=1).mean().item(),
            "payment_std": payments_epoch.sum(dim=1).std().item(),

            "unfairness_max": unfairness_epoch.max().item(),
            "unfairness_mean": unfairness_epoch.sum(dim=1).mean().item(),
            "unfairness_std": unfairness_epoch.sum(dim=1).mean().item(),

            "welfare": welfare_epoch.mean(dim=0).sum().item()
        }

        for key, value in train_stats.items():
            writer.add_scalar(f'train/{key}', value, global_step=epoch + args.resume_epoch)

        mult_stats = {
            "regret_mult": model.regret_mults.mean().item(),
            # "regret_rho": rho,
            "payment_mult": model.payment_mult,
            "fair_mult": model.fair_mults.mean().item(),
            # "fair_rho": rho_fair
        }

        for key, value in mult_stats.items():
            writer.add_scalar(f'multiplier/{key}', value, global_step=epoch + args.resume_epoch)

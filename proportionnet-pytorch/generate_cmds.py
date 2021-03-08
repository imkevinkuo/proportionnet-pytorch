from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('experiment', type=str, default='')
parser.add_argument('offset', type=float, default=0.0)

batch_size = 128
# batch_size = 4096
train_size = 640000

hps = {
    'num-examples': train_size,
    'test-num-examples': 10000,
    'lagr-update-iter': 100,
    # 'lagr-update-iter': 200,
    'rho-incr-amount': 1,
    'rho-incr-iter': 2 * train_size // batch_size,  # Every 2 epochs
    'lagr-update-iter-fair': 100,
    # 'lagr-update-iter-fair': 1000,
    'rho-incr-amount-fair': 1,
    'rho-incr-iter-fair': 2 * train_size // batch_size,
    'model-lr': 0.001,
    'misreport-lr': 0.1,
    'test-batch-size': batch_size,
    'batch-size': batch_size,
}


def script_2x2_opp():
    with open('2x2-opp.sh', 'w') as f:
        for seed in range(2):
            # Celis
            for c in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                f.write(f"python train.py")
                f.write(f" --dataset 2x2-opp")
                f.write(f" --random-seed {seed}")
                f.write(f" --num-epochs 120")
                f.write(f" --fairness cvg {c}")
                f.write(f" --name 2x2-opp_{seed}-s_{c}-c_40000-v")
                for k, v in hps.items():
                    f.write(f" --{k} {v}")
                f.write('\n')

            # for d in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            #     # Ilvento
            #     f.write(f"python train.py")
            #     f.write(f" --dataset 2x2-opp")
            #     f.write(f" --random-seed {seed}")
            #     f.write(f" --num-epochs 120")
            #     f.write(f" --fairness tvf {d}")
            #     f.write(f" --name 2x2-opp_{seed}-s_{d}-d")
            #     for k, v in hps.items():
            #         f.write(f" --{k} {v}")
            #     f.write('\n')

def script_1x2_mv():
    with open('1x2-mv_train.sh', 'w') as f:
        for i in range(10, 11):
            d = format(i*0.1, '.2f')
            for seed in range(19, 20):
                f.write(f"python train.py")
                f.write(f" --dataset 1x2-mv 1")
                f.write(f" --random-seed {seed}")
                f.write(f" --num-epochs 80")
                f.write(f" --fairness tvf {d}")
                f.write(f" --name 1x2-mv_{seed}-s_{d}-d")
                for k, v in hps.items():
                    f.write(f" --{k} {v}")
                f.write('\n')


def script_1x2_pv(offset):
    with open('1x2-pv_train.sh', 'w') as f:
        for i in range(11):
            d = format(i*0.1, '.2f')
            for seed in range(8):
                f.write(f"python train.py")
                f.write(f" --dataset 1x2-pv {offset}")
                f.write(f" --random-seed {seed}")
                f.write(f" --num-epochs 120")
                f.write(f" --fairness tvf {d}")
                f.write(f" --name 1x2-pv_{seed}-s_{d}-d")
                for k, v in hps.items():
                    f.write(f" --{k} {v}")
                f.write('\n')


# agent, item, layer, seed
# configs = [
#     (4, 6, 5, 0),
#     (4, 6, 4, 0),
# ]


def script_sweep_init():
    with open('mv_init.sh', 'w') as f:
        for n in range(5, 6):
            for m in range(2, 7):
        # for n, m, layers, seed in configs:
                layers = 2 + int(n * m / 10)
                # layers = min(5, 1 + int(n * m / 10))
                for seed in range(4, 6):
                    f.write(f"python train.py")
                    f.write(f" --dataset mv 1")
                    f.write(f" --random-seed {seed}")
                    f.write(f" --n-agents {n}")
                    f.write(f" --n-items {m}")
                    f.write(f" --n-hidden-layers {layers}")
                    f.write(f" --num-epochs 120")
                    f.write(f" --name {n}x{m}-mv_{seed}-s")
                    hps['rho-incr-amount-fair'] = 0
                    for k, v in hps.items():
                        f.write(f" --{k} {v}")
                    f.write('\n')


# configs = [
#     (4, 2, 2),
#     (4, 3, 3),
#     (4, 4, 3),
#     (4, 5, 4),
# ]

# configs = [
    # nm, layer, seed, resume
    # (5, 2, 3, 0, 119),
    # (5, 3, 3, 0, 119),
    # (5, 4, 4, 3, 119),
    # (5, 5, 4, 2, 119),
    # (5, 6, 5, 2, 119)
    # (5, 6, 5, 0, 119),
    # (5, 6, 5, 1, 119),
    # (5, 6, 5, 2, 119),
    # (5, 5, 4, 0, 119),
    # (5, 5, 4, 1, 119),
    # (5, 5, 4, 2, 119),
# ]


def script_sweep():
    with open('mv_sweep.sh', 'w') as f:
        # for n, m, layers, seed, r in configs:
        #     for z in range(1):
        for n in range(3, 5):
            for m in range(2, 7):
                seed = 0
                layers = min(5, 2 + int(n * m / 10))
                # for i in range(4):
                for i in range(4, 5):
                    d = format(i*0.25, '.2f')
                    f.write(f"python train.py")
                    f.write(f" --dataset mv 1")
                    f.write(f" --random-seed {seed}")
                    f.write(f" --n-agents {n}")
                    f.write(f" --n-items {m}")
                    f.write(f" --n-hidden-layers {layers}")
                    f.write(f" --name {n}x{m}-mv_{seed}-s_{d}-d")
                    # f.write(f" --resume result/{n}x{m}-mv_{seed}-s/{r}_checkpoint.pt")
                    f.write(f" --fairness tvf {d}")
                    f.write(f" --num-epochs 120")
                    f.write(f" --fair-full 120")
                    for k, v in hps.items():
                        f.write(f" --{k} {v}")
                    f.write('\n')

# 
# def script_fair_init():
#     with open('fair_init.sh', 'w') as f:
#         n, m = 3, 4
#         for b in range(5):
#             offset = format(b*0.25, '.2f')
#             layers = 3
#             for seed in range(1):
#                 f.write(f"python train.py")
#                 f.write(f" --dataset fr {offset}")
#                 f.write(f" --random-seed {seed}")
#                 f.write(f" --n-agents {n}")
#                 f.write(f" --n-items {m}")
#                 f.write(f" --n-hidden-layers {layers}")
#                 f.write(f" --name 3x4-fr_{offset}-b_{seed}-s")
#                 f.write(f" --num-epochs 120")
#                 for k, v in hps.items():
#                     f.write(f" --{k} {v}")
#                 f.write('\n')
# 
# 
# def script_frc_init():
#     with open('frc_init.sh', 'w') as f:
#         n, m = 3, 4
#         for b in range(5):
#             offset = format(b*0.25, '.2f')
#             layers = 3
#             for seed in range(1):
#                 f.write(f"python train.py")
#                 f.write(f" --dataset frc {offset}")
#                 f.write(f" --random-seed {seed}")
#                 f.write(f" --n-agents {n}")
#                 f.write(f" --n-items {m}")
#                 f.write(f" --n-hidden-layers {layers}")
#                 f.write(f" --name 3x4-frc_{offset}-b_{seed}-s")
#                 f.write(f" --num-epochs 120")
#                 for k, v in hps.items():
#                     f.write(f" --{k} {v}")
#                 f.write('\n')
# 
# 

def script_fair(args):
    if args.experiment == 'fair1_sweep':
        fairness = 'fr1'
    elif args.experiment == 'fair2_sweep':
        fairness = 'fr2'
    elif args.experiment == 'fair3_sweep':
        fairness = 'fr3'
    else:
        exit()

    with open(f'{args.experiment}.sh', 'w') as f:
        n = 3
        m = 4
        for b in range(5):
            offset = format(b*0.25, '.2f')
            layers = 3
            for i in range(4, 5):
                d = format(i*0.25, '.2f')
                for i in range(1):
                    for seed in range(1):
                        f.write(f"python train.py")
                        f.write(f" --dataset fr {offset}")
                        f.write(f" --random-seed {seed}")
                        f.write(f" --n-agents {n}")
                        f.write(f" --n-items {m}")
                        f.write(f" --n-hidden-layers {layers}")
                        f.write(f" --name 3x4-{fairness}_{offset}-b_{seed}-s_{d}-d")
                        f.write(f" --fairness tvf {fairness} {d}")
                        f.write(f" --num-epochs 120")
                        f.write(f" --fair-full 120")
                        # f.write(f" --resume ../backup/fr123/result/3x4-fr_{offset}-b_{seed}-s/119_checkpoint.pt")
                        for k, v in hps.items():
                            f.write(f" --{k} {v}")
                        f.write('\n')


import os
def script_test():
    with open("test.sh", 'w') as f:
        for s in range(10,14):
            for d in range(11):
                name = f"result/1x2-mv_{s}-s_{d/10:.2f}-d/90_checkpoint.pt"
                if os.path.exists("../" + name):
                    f.write(f"python test.py --name {name} --save-all\n")
        # for i in range(11):
        #     for s in range(8):
        #         f.write(f"python test.py --name ../backup/1x2-mv/result_export/1x2-mv_1_{i*0.1:.1f}_{s}.pt --save-all\n")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.experiment == 'mv':
        script_1x2_mv()
    if args.experiment == 'pv':
        script_1x2_pv(args.offset)
    if args.experiment == 'mv_init':
        script_sweep_init()
    if args.experiment == 'mv_sweep':
        script_sweep()
    if args.experiment == 'fair_init':
        script_fair_init()
    if args.experiment in ['fair1_sweep', 'fair2_sweep', 'fair3_sweep']:
        script_fair(args)
    if args.experiment == 'frc_init':
        script_frc_init()
    if args.experiment == 'test':
        script_test()
    if args.experiment == 'opp':
        script_2x2_opp()

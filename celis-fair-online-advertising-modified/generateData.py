import os

N_SAMPLES = 10000

if not os.path.exists("Webscope_A1"):
    os.mkdir("Webscope_A1")
with open("Webscope_A1/ydata-ysm-advertiser-bids-v1_0.txt", 'w') as f:
    for i in range(N_SAMPLES):
        # Agent 1: [1,2] x [2,3]
        f.write(f"06/15/2002 00:00:00\t0\t0\t{1 + i/N_SAMPLES:.4f}\t0\n")
        f.write(f"06/15/2002 00:00:00\t1\t0\t{2 + i/N_SAMPLES:.4f}\t0\n")
        # Agent 2: [2,3] x [1,2]
        f.write(f"06/15/2002 00:00:00\t0\t1\t{2 + i/N_SAMPLES:.4f}\t0\n")
        f.write(f"06/15/2002 00:00:00\t1\t1\t{1 + i/N_SAMPLES:.4f}\t0\n")

import os
import shutil
import re
import subprocess
from tqdm import tqdm
import time

# Specify a base file for all hyperparams
base_file = input("Please specify filepath of base settings:\n")
if base_file == "conf/global_settings.py":
    raise Exception("Choose different filepath for base settings. conf/global_settings.py is used by sweep to queue experiments.")
if not os.path.isfile(base_file):
    raise Exception("Invalid filepath.")

grid = {}
while True:
    hyperparam = input("\nSpecify hyperparam to vary (e.g. batch_size, lr, weight_decay) or 'END':\n")
    if hyperparam == "END": break
    grid[hyperparam] = []

    print("\n\tSpecify value of hyperparam or 'END':")
    while True:
        value = input("\t")
        if value == "END": break
        grid[hyperparam].append(value)

# Output and queue the experiments.
hpms = list(grid) # list of hyperparams
count = [0]*len(grid)

# confirm intent to queue experiments
num_jobs = 1
for hp in hpms:
    num_jobs *= len(grid[hp])
confirm = input("\nYou are about to queue " + str(num_jobs) + " jobs. Would you like to proceed? (y/n)\n")
if confirm != "y":
    raise Exception("User failed to confirm.")

# iterate through all hyperparams
log = []
iterate_loop = True
pbar = tqdm(total=num_jobs)
while iterate_loop:
    shutil.copyfile(base_file, 'conf/global_settings.py')

    for i, hp in enumerate(hpms):
        with open('conf/global_settings.py', 'r') as f:
            lines = f.readlines()

        for j in range(len(lines)):
            x = re.search(hp + " = ", lines[j])
            if x is not None and x.start() == 0:
                lines[j] = hp + " = " + grid[hp][count[i]] + "\n"

        with open('conf/global_settings.py', 'w') as f:
            for line in lines:
                f.write(line)
    
    # queue the experiment
    result = subprocess.run(["sh", "train.sh"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # log the experiment xid and hyperparams
    xids = subprocess.run(["ls", "-v", "results/"], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
    xid = xids[-2]
    
    log.append("xid: " + xid)
    log.append(result.stderr)
    for i, hp in enumerate(hpms):
        log.append("\t" + hp + "=" + grid[hp][count[i]])
    log.append("")

    # counting algorithm
    digit = len(grid)-1 # 0-indexed
    while True:
        if count[digit] < len(grid[hpms[digit]])-1:
            count[digit] += 1
            break
        else:
            count[digit] = 0
            digit -= 1
        if digit == -1:
            iterate_loop = False
            break
    
    time.sleep(0.5)
    pbar.update(1)

# write the log
with open('out_sweep.txt', 'w') as f:
    for line in log:
        f.write(line + "\n")
        print(line)
    

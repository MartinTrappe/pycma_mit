#!/usr/bin/env python3
"""
pycma_mit.py

Multi-run CMA-ES optimization driver using the Python `cma` package.
Runs up to N parallel CMA‐ES instances, tracks best‐of‐generation,
and saves results/plots in a timestamped `data/` folder.

Author: Martin-Isbjörn Trappe
Email: martin.trappe@quantumlah.org
Date: 2025-07-04
License: MIT License

Usage:
    ./run_pycma_mit.sh
    [sets up a virtual environment, installs the necessary dependencies and executes this script (pycma_mit.py)]

Dependencies:
    cma, numpy, matplotlib, pandas

See ===== BEGIN USER INPUT ===== END USER INPUT ===== below to adapt script for your objective
"""


import os
import math
import numpy as np
import subprocess
import concurrent.futures
import cma
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import time
from datetime import datetime
import shutil


# ===== BEGIN USER INPUT =====
threads = os.cpu_count()                       # number of threads for parallel evaluations
runs = max(1, threads)                         # number of independent CMA-ES runs
DIM = 32                                       # problem dimension
sigma0 = 0.3                                   # initial global step size (sigma), default 0.3, increase to 1.0 for more exploration
popsize = 7                                    # λ: offspring population size per generation
mu = popsize // 2                              # μ: number of parents for recombination
maxGeneration = 10                           # maximum number of generations
ReinflateSigma = False                         # if True, reinflate sigma when stagnating
diagDecoding = 1.0                             # diagonal→full covariance transition speed (0=slow,1=instant)
elitismQ = True                               # if True, always keep best parent each generation
ResetSchedule = 0#max(1, maxGeneration // runs)  # generations between killing worst run (0 to disable)

# === Objective functions ===

## Objective function: f = x^2
bounds = [[-1] * DIM, [1] * DIM] # domain
minimum = 0.0 # known true minimum
minimumAcc = 1.0e-5 # relative‐error tolerance for target encounters
def objective(x):
    return x*x

## Objective function: f = QuantumCircuitIA
# bounds = [[None] * DIM, [None] * DIM] # domain
# minimum = -5.770919159 # known true minimum
# minimumAcc = 1.0e-5 # relative‐error tolerance for target encounters
# def objective(x):
#     """
#     Run the external noisy_mps_vector_sim script with input vector x
#     and return its single float output. Penalize failures with +inf.
#     """
#     x_strs = [str(xi) for xi in x]               # convert each x[i] to string
#     cmd = ["./pycma_mit_scripts/noisy_mps_vector_sim-Martin.py"] + x_strs
#     try:
#         result = subprocess.run(
#             cmd, check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True
#         )
#         return float(result.stdout.strip())
#     except subprocess.CalledProcessError as e:
#         print("External script failed:", e.stderr)
#         return np.inf
#     except ValueError:
#         print("Could not parse float from output:", result.stdout)
#         return np.inf

# ===== END USER INPUT =====


script_start_time = time.time()       # record script start


# === Output directory and backup setup ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(script_dir, "data")
os.makedirs(output_dir, exist_ok=True)         # create data folder if missing

# Copy this script into data folder for reproducibility
script_path = os.path.realpath(__file__)
backup_name = f"{os.path.splitext(os.path.basename(script_path))[0]}_{timestamp}_backup.py"
shutil.copy(script_path, os.path.join(output_dir, backup_name))

# === Function to run a single CMA-ES instance ===
def run_cma_es(seed):
    """
    Not used in synchronized loop; returns (f_history, x_history, eval_count).
    """
    np.random.seed(seed)
    x0 = [0.0] * DIM                            # initial mean

    opts = {
        # === Population & recombination ===
        'popsize': popsize,                     # λ: offspring/population size
        'CMA_mu': mu,                           # μ: number of parents
        'CMA_active': True,                     # include negative weights (active CMA)
        'CMA_diagonal': True,                   # start with diagonal covariance
        'CMA_diagonal_decoding': diagDecoding,  # soft→full covariance transition rate

        # === Bound constraints ===
        'bounds': bounds,                       # box constraints: [lower], [upper]

        # === Elitism ===
        'CMA_elitist': elitismQ,                # keep best parent each generation

        # === Termination criteria ===
        'maxiter': maxGeneration,               # max number of generations
        'tolfun': 1e-12,                        # stop if f change < tolfun
        'tolfunhist': 1e-12,                    # strict history-based f change
        'tolx': 1e-12,                          # stop if x change < tolx
        'tolstagnation': maxGeneration // 10,   # stop after this many stagnations
        'tolflatfitness': maxGeneration // 10,  # additional flat-fitness stop

        # === Logging & display ===
        'verbose': 1,                           # internal verbosity (warnings/logs)
        'verb_disp': 1,                         # print summary every generation
        'verb_log': 1,                          # write .dat log files
        'verb_plot': 0                          # live plotting (requires matplotlib)
    }

    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    generation = 0
    best_f_history = []
    best_history = []
    xbest_history = []
    window_size = max(3, int(0.05 * maxGeneration))
    run_best_f = np.inf

    # Parallel ask–tell loop
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as pool:
        while not es.stop():
            generation += 1

            # Ask for candidate solutions
            solutions = es.ask()
            # Evaluate in parallel
            futures = [pool.submit(objective, x) for x in solutions]
            results = [f.result() for f in futures]

            # Tell CMA-ES the results
            es.tell(solutions, results)

            # Record histories
            best_f_history.append(es.best.f)
            xbest_history.append(es.best.x.copy())

            # Print only on improvement
            if es.best.f < run_best_f:
                run_best_f = es.best.f
                print(f"\n--- Generation {generation} ---")
                print(f"New local best f: {run_best_f:.6e}")
            else:
                print(".", end="", flush=True)

            # Optional sigma reinflation if stagnating
            if ReinflateSigma:
                best_history.append(es.best.f)
                if len(best_history) > window_size:
                    best_history.pop(0)
                    var_recent = np.var(best_history[-window_size:])
                    if var_recent < 1e-3 and es.sigma < 1e-1:
                        print(f"Re-inflating sigma from {es.sigma:.2e}")
                        es.sigma *= 10

    return best_f_history, xbest_history, es.result.evaluations


# === Synchronized multi-run CMA-ES ===

# 1) Initialize CMA-ES instances
es_list        = []
best_f_history = [[] for _ in range(runs)]
xbest_history  = [[] for _ in range(runs)]
active_runs    = list(range(runs))
eval_counts    = [0] * runs

# counters for how many runs have ever hit the target
target_encounters = 0
hit_runs = set()    # to remember which run-IDs have already been counted

for rid in range(runs):
    x0 = [0.0] * DIM
    opts = {
        'popsize': popsize,
        'CMA_mu': mu,
        'CMA_active': True,
        'CMA_diagonal': True,
        'CMA_diagonal_decoding': diagDecoding,
        'CMA_elitist': elitismQ,
        'bounds': bounds,
        'maxiter': maxGeneration,
        'tolfun': 1e-12,
        'tolfunhist': 1e-12,
        'tolx': 1e-12,
        'tolstagnation': maxGeneration // 10,
        'tolflatfitness': maxGeneration // 10,
        'verbose': 1, 'verb_disp': 1, 'verb_log': 1, 'verb_plot': 0
    }
    es_list.append(cma.CMAEvolutionStrategy(x0, sigma0, opts))

# 2) Create thread pool for generation-synchronized evaluations
pool = concurrent.futures.ThreadPoolExecutor(max_workers=threads)

# 3) Main generation loop
for generation in range(1, maxGeneration + 1):
    # a) Ask each active run for offspring
    asks = [es_list[rid].ask() for rid in active_runs]
    # b) Flatten all solutions
    flat_sols = [x for sub in asks for x in sub]
    # c) Evaluate all in parallel
    flat_vals = list(pool.map(objective, flat_sols))

    # d) Distribute results back to each CMA-ES
    idx = 0
    for j, rid in enumerate(active_runs):
        lam = es_list[rid].opts['popsize']
        sub_vals = flat_vals[idx:idx + lam]
        idx += lam
        es_list[rid].tell(asks[j], sub_vals)
        best_f_history[rid].append(es_list[rid].best.f)
        xbest_history[rid].append(es_list[rid].best.x.copy())
        eval_counts[rid] += lam

    # e) Print global best so far
    current_global_best = min(best_f_history[rid][-1] for rid in active_runs)
    print(f"[Global] Generation {generation}, best f across runs = {current_global_best:.6e}")

    # Count any new “target encounters” this generation
    for rid in active_runs:
        if rid not in hit_runs:
            a = best_f_history[rid][-1]   # current best f for run rid
            b = minimum
            if b == 0.0:
                rel = rel = abs(a)
            else:
                den = max(abs(a), abs(b))
                if den > 0.0:
                    rel = abs(a - b) / den
            if rel < minimumAcc:
                target_encounters += 1
                hit_runs.add(rid)
                print(f"[Target] Run {rid} reached target at gen {generation} "
                      f"(f={a:.6e}, rel_diff={rel:.2e})")

    # f) Optionally kill worst run according to ResetSchedule
    if ResetSchedule and generation % ResetSchedule == 0 and len(active_runs) > 1:
        worst = max(active_runs, key=lambda rid: best_f_history[rid][-1])
        print(f"\n[ResetSchedule] Killing worst run {worst} "
              f"(f = {best_f_history[worst][-1]:.6e})")
        active_runs.remove(worst)

# 4) Shutdown the pool
pool.shutdown()

# === Pad & aggregate results ===

# Pad f-histories to common length
max_len = max(len(hist) for hist in best_f_history)
padded = np.array([
    hist + [hist[-1]] * (max_len - len(hist))
    for hist in best_f_history
])

# Pad x-histories similarly
x_padded = []
for hist_x in xbest_history:
    last = hist_x[-1]
    x_padded.append(hist_x + [last] * (max_len - len(hist_x)))
x_padded = np.array(x_padded)  # shape: (runs, max_len, DIM)

# Compute per-generation global best f and corresponding x
global_best_f = np.min(padded, axis=0)
best_run_idx  = np.argmin(padded, axis=0)
global_best_x = np.array([x_padded[best_run_idx[i], i] for i in range(max_len)])

# === Save global best trajectory to file ===
txt_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(script_path))[0]}_{timestamp}_global_best_trajectory.dat")
np.savetxt(
    txt_path,
    np.column_stack([global_best_f, global_best_x]),
    fmt="%.10f",
    delimiter=" "
)
print(f"Saved f(x) and x to {os.path.splitext(os.path.basename(script_path))[0]}_{timestamp}_global_best_trajectory.dat")

# === Print final global best ===
final_best_f      = global_best_f[-1]
final_best_x      = global_best_x[-1]
total_evaluations = sum(eval_counts)
print("\n=== Global Optimization Result ===")
print(f"\nTotal runs that ever reached the target within ±{minimumAcc:.0e} relative accuracy: {target_encounters}/{runs}")
print(f"Total function evaluations: {total_evaluations}")
print(f"Global best f(x): {final_best_f:.6e}")
print("Global best x:", final_best_x)

# === Total runtime ===
script_end_time = time.time()
total_secs = script_end_time - script_start_time
print(f"\nTotal wall-clock time: {total_secs:.2f} seconds")

# === Plotting ===
plt.figure(figsize=(10, 6))

# Select top-10 runs by final f
final_vals  = np.array([hist[-1] for hist in padded])
top_indices = np.argsort(final_vals)[:10]
colors      = plt.get_cmap("tab10")

# Plot all runs: light gray for non-top, strong colors for top-10
for i, hist in enumerate(padded):
    if i in top_indices:
        c   = colors(list(top_indices).index(i))
        lbl = f"Run {i+1} (f={hist[-1]:.6e})"
        plt.plot(hist, color=c, alpha=0.9, linewidth=1.5, label=lbl)
    else:
        plt.plot(hist, color="#cccccc", alpha=0.5, linewidth=1.0)

# Plot global best trajectory as dotted black line
plt.plot(
    global_best_f,
    color='black',
    linewidth=2.5,
    linestyle=':',
    label="Global Best f(x)"
)

plt.xlabel("Generation")
plt.ylabel("Best f(x)")
plt.title(
    f"pycma_mit.py\n {runs} runs --- {maxGeneration} generations --- {math.floor(total_secs)} sec:\n"
    f"best f = {final_best_f:.6e}  @  #FuncEvals = {total_evaluations}   #TargetEncounters = {target_encounters} / {runs}"
)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
plot_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(script_path))[0]}_{timestamp}_global_plot.pdf")
plt.savefig(plot_path)
plt.show()

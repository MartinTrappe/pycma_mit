#!/usr/bin/env python3
"""
pycma_mit.py

Multi-run CMA-ES optimization driver using the Python `cma` package.
Runs up to N parallel CMA‐ES instances, tracks best‐of‐generation,
and saves results/plots in a timestamped `data/` folder.

Author: Martin-Isbjörn Trappe
Email: martin.trappe@quantumlah.org
Date: 2025-07-04
License: License

Usage:
    ./run_pycma_mit.sh
    [sets up a virtual environment, installs the necessary dependencies and executes this script (pycma_mit.py)]

Dependencies:
    cma, numpy, matplotlib, pandas

See ===== BEGIN USER INPUT ===== below to adapt script for your objective
"""


import os
import math
import numpy as np
import subprocess
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import cma
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import time
from datetime import datetime
import shutil
from scipy.optimize import minimize


# ============================
# ===== BEGIN USER INPUT =====
# ============================
# CHOOSE OBJECTIVE FUNCTION (add definitions below) to run:
#   'quadratic'  →  f(x)=x²
#   'QuantumCircuitIA'    →  external noisy_mps_vector_sim-Martin
func_id      = 'QuantumCircuitIA'
# CHOOSE OPTIMIZER:
# Covariance Matrix Adaptation Evolution Strategy (CMA-ES, main purpose of this driver):
# 'cma'
# Derivative-free constrained methods:
#   'COBYLA', 'COBYQA'
# Bound-constrained only:
#   'L-BFGS-B', 'TNC'
# Gradient-based constrained methods:
#   'SLSQP', 'trust-constr'
# Unconstrained / bound-free or bound-constrained algorithms:
#   'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
#   'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'
# Custom:
#   Any user-supplied callable implementing the same signature
#   as the built-in methods for scipy.optimize.minimize (e.g., fun(x, *args), jac(x, *args), etc.)
OPTIMIZER = 'cma'
threads = os.cpu_count()                       # number of threads for parallel evaluations
runs = max(1, threads)                         # number of independent CMA-ES runs
DIM = 20                                       # problem dimension
sigma0 = 0.3                                   # initial global step size (sigma), default 0.3, increase to 1.0 for more exploration
popsize = 7                                    # λ: offspring population size per generation
mu = popsize // 2                              # μ: number of parents for recombination
maxGeneration = 200                            # maximum number of generations
diagDecoding = 1.0                             # diagonal→full covariance transition speed (0=slow,1=instant)
elitismQ = True                                # if True, always keep best parent each generation
killQ = 0    # generations between killing worst run (default: max(1, maxGeneration // runs); 0 to disable; max(1, maxGeneration // (2*runs)) for fast decay to one remaining run)
# ==========================
# ===== END USER INPUT =====
# ==========================


script_start_time = time.time()       # record script start


# Lower this process’s priority on Linux
try:
    subprocess.run(['renice', '+19', '-p', str(os.getpid())],
                   check=True, stdout=subprocess.DEVNULL)
except Exception:
    pass  # if renice fails, we still proceed


# Shared global best and lock
global_best = float('inf')
best_lock   = threading.Lock()


if func_id == 'quadratic':
    # f(x)=x^2 on [-1,1]^DIM
    bounds      = [[-1]*DIM, [1]*DIM]
    minimum     = 0.0
    minimumAcc  = 1e-5
    def objective(x):
        return float(np.dot(x, x))

elif func_id == 'QuantumCircuitIA':
    # external noisy_mps_vector_sim script with input vector x and return its single float output. Penalize failures with +inf.
    DMPEPS = True # use Density-Matrix Circuit
    #bounds      = [[None]*DIM, [None]*DIM]
    bounds      = [[0.0]*DIM, [2*np.pi]*DIM] # seems to work better
    minimum     = -5.770919159
    minimumAcc  = 1e-5
    def objective(x):
        x_strs = [str(xi) for xi in x]               # convert each x[i] to string
        if DMPEPS:
            cmd = ["./pycma_mit_scripts/noisy-DM-PEPS-sim.py", "eval"] + x_strs
        else:
            cmd = ["./pycma_mit_scripts/noisy_mps_vector_sim-Martin.py"] + x_strs
        try:
            result = subprocess.run(
                cmd, check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return float(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print("External script failed:", e.stderr)
            return np.inf
        except ValueError:
            print("Could not parse float from output:", result.stdout)
            return np.inf

# add more func_id here

else:
    raise ValueError(f"Unknown func_id: {func_id}")


# Derived input parameters
budget = runs * maxGeneration * popsize
print(f"    #threads = {threads}")
print(f"Total Budget = {budget}")
lower, upper = bounds


def random_x0(lower, upper, sigma0):
    x0 = []
    for lo, hi in zip(lower, upper):
        if lo is not None and hi is not None:
            # both finite → uniform in [lo, hi]
            x0.append(np.random.uniform(lo, hi))
        else:
            # unbounded on at least one side → Gaussian fallback
            x0.append(np.random.normal(0.0, sigma0))
    return x0


def run_cma():
    # centralized CMA-ES options:
    cma_opts = {
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


    # === Output directory and backup setup ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)         # create data folder if missing


    # Copy this script into data folder for reproducibility
    script_path = os.path.realpath(__file__)
    backup_name = f"{os.path.splitext(os.path.basename(script_path))[0]}_{timestamp}_backup.py"
    shutil.copy(script_path, os.path.join(output_dir, backup_name))



    # =====================================
    # === Synchronized multi-run CMA-ES ===
    # =====================================

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
        opts = dict(cma_opts)
        #x0 = [0.0]*DIM
        x0 = random_x0(lower, upper, sigma0)
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
                    print(f"[Target] Run {rid} reached target at gen {generation} (f={a:.6e}, rel_diff={rel:.2e})")

        # f) Optionally kill worst run according to killQ
        if killQ and generation > maxGeneration // 4 and generation % killQ == 0 and len(active_runs) > 1:
            worst = max(active_runs, key=lambda rid: best_f_history[rid][-1])
            print(f"\n[Termination] Killing worst run {worst} (f = {best_f_history[worst][-1]:.6e})")
            active_runs.remove(worst)

    # 4) Shutdown the pool
    pool.shutdown()


    # ==========================================
    # === Pad, aggregate, and export results ===
    # ==========================================

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


def run_scipy_once(method, seed):
    np.random.seed(seed)
    x0 = random_x0(lower, upper, sigma0)
    cum_evals = 0

    def fun(x):
        nonlocal cum_evals
        f = objective(x)
        cum_evals += 1
        # 3) print only if f beats the global best so far
        with best_lock:
            global global_best
            if f < global_best:
                global_best = f
                print(
                    f"[Seed {seed}]:  local evals = {cum_evals} — best f across runs = {f:.6e}",
                    flush=True
                )
        return f

    # COBYLA treats maxiter as max-func-evals; give it the full per-run budget
    if method.upper() == 'COBYLA':
        opts = {'maxiter': max(DIM + 2, maxGeneration - (DIM + 2))}
    else:
        opts = {'maxiter': maxGeneration}

    res = minimize(
        fun,
        x0,
        method=method,
        bounds=list(zip(lower, upper)),
        options=opts
    )

    return {
        'seed':       seed,
        'final_f':    res.fun,
        'total_nfev': cum_evals,
        'best_x':     res.x,
        'message':    res.message
    }

if __name__ == '__main__':
    if OPTIMIZER.lower() == 'cma':
        run_cma()
    else:
        if OPTIMIZER == 'COBYLA':
            n_runs = budget // max(DIM + 2, maxGeneration - (DIM + 2))
        elif OPTIMIZER == 'Nelder-Mead':
            n_runs = budget // (2*maxGeneration)
        else:
            n_runs = runs
        print(f"Launching {n_runs} runs of {OPTIMIZER} with a budget of ~{budget//n_runs} func evals each across all iterations\n")

        seeds  = list(np.random.randint(0, 1_000_000, size=n_runs))
        results = []
        with ThreadPoolExecutor(max_workers=n_runs) as ex:
            futures = {ex.submit(run_scipy_once, OPTIMIZER, s): s for s in seeds}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                print(
                    f"\n=== Seed {r['seed']} done ===\n"
                    f"   final f(x):   {r['final_f']:.6e}\n"
                    f"  #Iterations:   {r['total_nfev']}\n"
                    f"      message:   {r['message']}\n"
                )

        best = min(results, key=lambda r: r['final_f'])
        print("=== Best of all runs ===")
        print(f"       Seed: {best['seed']}")
        print(f"  Best f(x): {best['final_f']:.6e}")
        print(f"#Iterations: {best['total_nfev']}")
        print(f"     Best x: {best['best_x']}")


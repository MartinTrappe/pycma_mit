#!/usr/bin/env python3
"""
pycma_mit.py

Multi-run CMA-ES optimization driver using the Python `cma` package.
Runs up to N parallel CMA‐ES instances, tracks best‐of‐generation,
and saves results/plots in a timestamped `data/` folder. Fair comparison
with all scipy optimizers. COBYLA, for example, is sequential, and if
we cannot afford a lot of generations, then we compensate with many
parallel runs.

Author: Martin-Isbjörn Trappe
Email: martin.trappe@quantumlah.org
Date: 2025-07-17
License: License

Usage:
    ./run_pycma_mit.sh
    [sets up a virtual environment, installs the necessary dependencies and executes this script (pycma_mit.py)]

Dependencies:
    cma, numpy, matplotlib, pandas

See ===== BEGIN USER INPUT ===== below to adapt script for your objective
"""


import sys, os
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
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# ============================
# ===== BEGIN USER INPUT =====
# ============================
# CHOOSE OBJECTIVE FUNCTION (add definitions below):
#   'quadratic'  →  f(x)=x²
#   'QuantumCircuitIA_1' & 'QuantumCircuitIA_2'    →  external scripts located in pycma_mit_scripts/
func_id      = 'quadratic'
# CHOOSE OPTIMIZER (see selection below):
OPTIMIZER = 'Adam'
gradQ = True                                  # True if scipy should use gradient (user-supplied in objective function); False for finite-difference
autogradQ = False                              # True to use automatic differentiation (ToDo)
threads = 10#os.cpu_count() - 2                   # number of threads for parallel evaluations
runs = 20                                      # number of independent CMA-ES runs
DIM = 20                                       # problem dimension
sigma0 = 0.3                                   # initial global step size (sigma), default 0.3, increase to 1.0 for more exploration
popsize = 6                                    # λ: offspring population size per generation
mu = popsize // 2                              # μ: number of parents for recombination
maxGeneration = 400                            # maximum number of generations
diagDecoding = 1.0                             # diagonal→full covariance transition speed (0=slow,1=instant)
elitismQ = True                                # if True, always keep best parent each generation
killQ = 0                                      # cma-generations between killing worst run (default: max(1, maxGeneration // runs); 0 to disable; max(1, maxGeneration // (2*runs)) for fast decay to one remaining run)
# ==========================
# ===== END USER INPUT =====
# ==========================


# Derived input parameters

# OPTIMIZER selection
# 1) Gradient-based methods
#  1.a  Bound-constrained ONLY
gradient_bound = [
    'L-BFGS-B',
    'TNC',
]
#  1.b  Unconstrained ONLY
gradient_unconstrained = [
    'CG',
    'BFGS',
    'Newton-CG',
    'dogleg',
    'trust-ncg',
    'trust-exact',
    'trust-krylov',
]
#  1.c  Unconstrained OR bound-constrained
gradient_either = [
    'SLSQP',
    'trust-constr',
    'Adam',
]
# 2) Derivative-free methods
#  2.a  Bound-constrained ONLY
derivative_free_bound = [
    'COBYLA',
    'COBYQA',
]
#  2.b  Unconstrained OR bound-constrained
derivative_free_either = [
    'cma', # Covariance Matrix Adaptation Evolution Strategy (CMA-ES, main purpose of this driver)
    'Nelder-Mead',
    'Powell',
]
if gradQ == True and ((OPTIMIZER in derivative_free_bound) or (OPTIMIZER in derivative_free_either)):
    gradQ = False
    print("!!! Warning: gradQ changed to False !!!")

if func_id == 'quadratic':
    # f(x)=x^2 on [-1,1]^DIM
    BOUNDS      = [[-1]*DIM, [1]*DIM]
    minimum     = 0.0
    minimumAcc  = 1e-5
    def objective(x):
        f = float(np.dot(x, x))
        if gradQ == True:
            g = 2*x
            return f, g
        else:
            return f

elif func_id == 'QuantumCircuitIA_1' or func_id == 'QuantumCircuitIA_2':
    # external noisy_mps_vector_sim script with input vector x and return its single float output. Penalize failures with +inf.
    #BOUNDS      = [[None]*DIM, [None]*DIM]
    BOUNDS      = [[0.0]*DIM, [2*np.pi]*DIM] # seems to work better
    minimum     = -5.770919159
    minimumAcc  = 1e-5
    def objective(x):
        x_strs = [str(xi) for xi in x]               # convert each x[i] to string
        if func_id == 'QuantumCircuitIA_1':
            cmd = ["./pycma_mit_scripts/noisy_mps_vector_sim-Martin.py"] + x_strs
        elif func_id == 'QuantumCircuitIA_2':
            cmd = ["./pycma_mit_scripts/noisy-DM-PEPS-sim.py", "eval"] + x_strs
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

# USER: add more func_id HERE
# elif func_id == '????':
    # return ?

else:
    raise ValueError(f"Unknown func_id: {func_id}")


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


# send all print()s and tracebacks to both console and log_file
script_base = os.path.splitext(os.path.basename(__file__))[0]
log_path    = os.path.join(output_dir, f"{script_base}_{timestamp}_maxGen={maxGeneration}_{OPTIMIZER}.log")
# open the logfile (line-buffered)
log_file = open(log_path, 'w', buffering=1)
# define a simple Tee
class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()
# wrap both stdout and stderr
sys.stdout = TeeStream(sys.__stdout__, log_file)
sys.stderr = TeeStream(sys.__stderr__, log_file)


# Lower this process’s priority on Linux
try:
    subprocess.run(['renice', '+19', '-p', str(os.getpid())],
                   check=True, stdout=subprocess.DEVNULL)
except Exception:
    pass  # if renice fails, we still proceed


# Shared global best and lock
global_best = float('inf')
best_lock   = threading.Lock()


# Derived input parameters
budget = runs * maxGeneration * popsize
print(f"    #threads = {threads}")
print(f"Total Budget = {budget}")
if OPTIMIZER in gradient_unconstrained:
    BOUNDS_scipy = None
    lower = [None]*DIM
    upper = [None]*DIM
    print("!!! Warning: BOUNDS changed to unconstrained !!!")
else:
    lower, upper = BOUNDS
    BOUNDS_scipy = list(zip(lower, upper))


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
        'bounds': BOUNDS,                       # box constraints: [lower], [upper]

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
        f"pycma_mit.py\n {runs} runs --- {maxGeneration} generations:\n"
        f"best f = {final_best_f:.6e}  @  #FuncEvals = {total_evaluations}   #TargetEncounters = {target_encounters} / {runs}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plot_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(script_path))[0]}_{timestamp}_global_plot.pdf")
    plt.savefig(plot_path)
    #plt.show()


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

    # COBYLA: although DIM+2 evals per run are added as preparation (which could, in principle, be done in parallel, so let's not count them), but it needs at least DIM+2 func evals, for which it asigns, apparently, the 'maxiter' option
    if method == 'COBYLA':
        opts = {'maxiter': maxGeneration + DIM + 2}
    elif method == 'TNC':
        opts = {
            'maxfun': maxGeneration*((DIM+1)*(10+1)+4),
            'disp': True
            }
    else:
        opts = {'maxiter': maxGeneration}

    res = minimize(
        fun,
        x0,
        method=method,
        jac=gradQ,              # True ⇒ fun_and_grad must return (f, g)
        bounds=BOUNDS_scipy,
        options=opts
    )

    return {
        'seed':       seed,
        'final_f':    res.fun,
        'total_nfev': cum_evals,
        'best_x':     res.x,
        'message':    res.message
    }


def run_adam_once(seed: int) -> dict:
    global global_best

    np.random.seed(seed)
    cum_evals = 0
    best_local = float('inf')

    # initial params
    params_np = np.array(random_x0(lower, upper, sigma0))
    params    = torch.tensor(params_np, dtype=torch.float64, requires_grad=False)

    # Adam optimizer
    optimizer = Adam([params],
                     lr=0.1,
                     betas=(0.9, 0.999),
                     eps=1e-8,
                     amsgrad=True)

    # scheduler: step down by factor=0.1 every 200 iters
    scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

    for gen in range(1, maxGeneration+1):
        optimizer.zero_grad()

        # compute f and g in NumPy
        if gradQ:
            f_val, g = objective(params_np)
            cum_evals += 1
        else:
            f_val = objective(params_np)
            cum_evals += 1
            g = np.zeros_like(params_np)
            eps = 1e-6
            for i in range(len(params_np)):
                x_eps = params_np.copy()
                x_eps[i] += eps
                cum_evals += 1
                g[i] = (objective(x_eps) - f_val) / eps

        # inject gradient and step
        params.grad = torch.tensor(g, dtype=torch.float64)
        optimizer.step()

        # advance the scheduler
        scheduler.step()

        # pull params back to NumPy
        params_np = params.detach().numpy()

        # global‐best logging
        with best_lock:
            if f_val < global_best:
                global_best = f_val
                print(f"[Seed {seed}]: evals={cum_evals} best={f_val:.6e}", flush=True)

        if f_val < best_local:
            best_local = f_val

    total_nfev = cum_evals
    return {
        'seed':       seed,
        'final_f':    best_local,
        'total_nfev': total_nfev,
        'best_x':     params_np.copy(),
        'message':    'completed'
    }




if __name__ == '__main__':
    if OPTIMIZER == 'cma':
        run_cma()
    else:
        if OPTIMIZER == 'COBYLA' or OPTIMIZER == 'Powell':
            evals_per_gen = 1
        elif OPTIMIZER == 'Nelder-Mead':
            evals_per_gen = 2
        elif OPTIMIZER == 'TNC':
            evals_per_gen = (DIM+1)*(10+1)+4
        elif OPTIMIZER == 'CG':
            evals_per_gen = (DIM+1)*2
        elif OPTIMIZER == 'Adam':
            if gradQ:
                evals_per_gen = 1
            else:
                evals_per_gen = DIM+1
        else:
            evals_per_gen = DIM+1
        n_runs = max(1,budget // (maxGeneration*evals_per_gen))
        print(f"Launching {n_runs} runs of {OPTIMIZER} with a budget of ~{budget//n_runs} func evals each across all {maxGeneration} iterations\n")

        seeds  = list(np.random.randint(0, 1_000_000, size=n_runs))
        results = []
        with ThreadPoolExecutor(max_workers=threads) as ex:
            if OPTIMIZER == 'Adam':
                futures = {ex.submit(run_adam_once, s): s for s in seeds}
            else:
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

    # === Total runtime ===
    script_end_time = time.time()
    total_secs = script_end_time - script_start_time
    print(f"\nTotal wall-clock time: {total_secs:.2f} seconds")


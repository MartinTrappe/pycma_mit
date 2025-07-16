# pycma_mit

Multi-run CMA-ES optimization driver (using the Python `cma` package from N. Hansen et al.)

## Overview

`pycma_mit` launches multiple parallel CMA-ES runs, tracks the best fitness value of each run, and saves timestamped logs and plots in the automatically created `data/` folder. Customizable hyperparemeters to improve performance for specific use cases.

**Key outputs:**

- data for best fitness trajectory
- log of all output
- PDF plot of optimization progress of all runs
- Automatic backup copy of the script at execution time

## Author

**Martin-Isbjörn Trappe**

Email: martin.trappe@quantumlah.org

July 4, 2025

## Features

- **Parallel execution:** run multiple CMA-ES instances concurrently
- **Customizable:** objective functions (optionally to be called from other scipts), population size, generation count, etc.
- **Elitism & Co:** built-in mechanisms to avoid premature (or accelerate) convergence
- **Logging & plotting:** auto-saved `.dat` files and Matplotlib figures
- **Easy integration:** template hooks for external programs
- **fair (same-budget) comparison**: option to run all scipy optimizers (COBYLA, Nelder-Mead, CG, etc.) instead of CMA-ES

## Prerequisites

- **Python** 3.7 or later
- **bash** shell (for `run_pycma_mit.sh`)

## Installation

**Clone** the repository:
   ```bash
   git clone https://github.com/MartinTrappe/pycma_mit.git
   cd pycma_mit
   ```

## File Structure

```
.
├── pycma_mit.py          # Main CMA-ES driver
├── run_pycma_mit.sh      # Setup & run wrapper
├── data/                 # Generated logs & plots (auto-created)
├── README.md             # This file
├── LICENSE               # License
└── .gitignore            # Excludes venv/, data/, __pycache__/, etc.
```

## Usage

**Edit** the `# === USER INPUT ===` block in `pycma_mit.py` to set your parameters:
   - Objective function
   - Search space dimension
   - Number of runs
   - Initial step-size (`sigma0`)
   - Population size (`popsize`)
   - Max generations (`maxGeneration`)
   - Elitism, reinflation periods, etc.

**Run** the setup and script:
   ```bash
   ./run_pycma_mit.sh
   ```

**Inspect** results in the newly created `data/` folder.


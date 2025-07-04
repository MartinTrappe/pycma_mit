# pycma_mit

Multi-run CMA-ES optimization driver (using the Python `cma` package from N. Hansen et al.)

## Overview

`pycma_mit` launches multiple parallel CMA-ES runs (via `concurrent.futures`), collects the best-of-generation fitness values, and saves logs and plots in a timestamped `data/` folder.

**Key outputs:**

- data for best fitness trajectory
- log of all output
- PDF plot of optimization progress of all runs
- Backup copy of the script

## Author

**Martin-Isbjörn Trappe**

Email: martin.trappe@quantumlah.org

July 4, 2025

## Features

- **Parallel execution:** run multiple CMA-ES instances concurrently
- **Customizable:** dimension, population size, sigma, generation count, etc.
- **Elitism & reinflation:** built-in mechanisms to avoid premature convergence
- **Logging & plotting:** auto-saved `.dat` files and Matplotlib figures
- **Easy integration:** template hooks for external programs

## Prerequisites

- **Python** 3.7 or later
- **bash** shell (for `run_pycma_mit.sh`)
- **Git** (for version control)

## Installation

**Clone** the repository:
   ```bash
   git clone https://github.com/<your-username>/pycma_mit.git
   cd pycma_mit
   ```

## File Structure

```
.
├── pycma_mit.py          # Main CMA-ES driver
├── run_pycma_mit.sh      # Setup & run wrapper
├── data/                 # Generated logs & plots (auto-created)
├── README.md             # This file
├── LICENSE               # MIT License
└── .gitignore            # Excludes venv/, data/, __pycache__/, etc.
```

## Usage

**Edit** the `# === USER INPUT ===` block in `pycma_mit.py` to set your parameters:
   - Objective function
   - Number of runs
   - Initial step-size (`sigma0`)
   - Population size (`popsize`)
   - Max generations (`maxGeneration`)
   - Elitism intervals, reinflation periods, etc.

**Run** the setup and script:
   ```bash
   ./run_pycma_mit.sh
   ```

**Inspect** results in the newly created `data/` folder.


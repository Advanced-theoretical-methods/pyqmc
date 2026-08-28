# pyqmc

Python library for real-space quantum Monte Carlo (QMC) built on [PySCF](https://pyscf.org). This repository extends the [Wagner Group pyqmc](https://github.com/WagnerGroup/pyqmc) codebase with **ABCDMC** (Auxiliary-field Boson Corrected Diffusion Monte Carlo) and related auxiliary-boson methods.

## Features

### Standard QMC (from upstream pyqmc)

- **Variational Monte Carlo (VMC)** and **Diffusion Monte Carlo (DMC)**
- Slater–Jastrow trial wavefunctions with PySCF integration (HF, CASSCF, etc.)
- Wavefunction optimization via line minimization and orthogonal excited-state optimization
- Periodic systems, ECPs, twist averaging, and one-/two-body density matrices
- High-level recipes (`OPTIMIZE`, `VMC`, `DMC`) and lower-level APIs for full control
- Parallel execution via `mpi4py` or `dask`

### ABCDMC extensions (this fork)

- **Auxiliary Boson** wavefunctions for multi-determinant and strongly correlated systems
- **ABOPTIMIZE** — optimize boson × Jastrow trial wavefunctions
- **ABVMC** — auxiliary-boson variational Monte Carlo
- **ABDMC** — auxiliary-boson diffusion Monte Carlo with the `abc_dmc_excitations` matrix accumulator for overlap and Hamiltonian matrix elements
- Optional wall-clock profiling for boson DMC (`profile_boson_dmc` in config or `PYQMC_PROFILE_*` environment variables)

## Installation

**Requirements:** Python 3.8+, [PySCF](https://pyscf.org), SciPy, pandas, h5py

```bash
pip install scipy pandas pyscf h5py
pip install git+https://github.com/Advanced-theoretical-methods/pyqmc.git
```

Or clone and install locally:

```bash
git clone https://github.com/Advanced-theoretical-methods/pyqmc.git
cd pyqmc
pip install -e .
```

For running tests that use the naive HCI module:

```bash
pip install "pyscf[naive_hci]"
```

**Optional dependencies**

| Package | Used for |
|---------|----------|
| `mpi4py` | MPI-parallel VMC/DMC and ABCDMC |
| `dask` | Distributed parallel workflows |
| `numba` | Faster ABCDMC matrix accumulation |
| `pyyaml` | YAML-driven production scripts (see examples) |

## Quick start

### ABCDMC workflow

After generating PySCF SCF and CI checkpoint files, use the boson recipes:

```python
from pyqmc import bosonrecipes

jastrow_kws = {"ion_cusp": False, "na": 1, "nb": 1, "rcut": 5}

# Optimize Jastrow factor
bosonrecipes.ABOPTIMIZE(
    "scf.hdf5",
    "opt.hdf5",
    ci_checkfile="ci.hdf5",
    jastrow_kws=jastrow_kws,
    det_emax="0.5,singles",
    use_symm=True,
)

# Run ABVMC
bosonrecipes.ABVMC(
    "scf.hdf5",
    "abvmc.hdf5",
    ci_checkfile="ci.hdf5",
    load_parameters="opt.hdf5",
    jastrow_kws=jastrow_kws,
    nblocks=100,
)

# Run ABDMC with matrix-element accumulation
bosonrecipes.ABDMC(
    "scf.hdf5",
    "abcdmc.hdf5",
    ci_checkfile="ci.hdf5",
    load_parameters="opt.hdf5",
    jastrow_kws=jastrow_kws,
    accumulators=["abc_dmc_excitations"],
    nblocks=5000,
    tstep=0.005,
)
```

Production ABCDMC examples for a Be⁺ cation live under [`examples/boson/`](examples/boson/):

| Directory | XC functional | Run directory |
|-----------|---------------|---------------|
| [`lda-be-cation/`](examples/boson/lda-be-cation/) | LDA (VWN) | [`dt-0.005/`](examples/boson/lda-be-cation/dt-0.005/) |
| [`pbe-be-cation/`](examples/boson/pbe-be-cation/) | PBE | [`dt-0.005/`](examples/boson/pbe-be-cation/dt-0.005/) |

Each example is driven by a YAML config (`config.yaml`) and MPI. Generate SCF/CI checkpoints with the scripts in `wfs/`, then run ABCDMC from the `dt-0.005/` directory:

```bash
cd examples/boson/pbe-be-cation/dt-0.005
# after wfs/*.hdf5 exist:
mpirun -n <N> python -u -m mpi4py.futures run.py
```

See [`job.sh`](examples/boson/pbe-be-cation/dt-0.005/job.sh) for a Slurm submission template.

#### `det_emax` — CI determinant selection

The auxiliary boson wavefunction is built from determinants taken from a PySCF CI expansion (CASSCF, HCI, etc.). `det_emax` controls **which determinants are kept** before the boson expansion is constructed. Filtering uses mean-field (sum of occupied MO energies) and excitation counts relative to the reference determinant.

| Form | Meaning |
|------|---------|
| `float` (e.g. `0.5`) | Keep determinants with MF energy ≤ reference + `float` (Hartree) |
| `int` 1–100 (e.g. `50`) | Keep determinants up to that percentile of MF energies |
| `"singles"` | Reference + single excitations only (`tot_exc < 2`) |
| `"doubles"` | Reference + singles + doubles (`tot_exc < 3`) |
| `"energy,criteria"` (e.g. `"0.5,singles"`) | **Both** an energy window **and** an excitation cap: keep determinants with MF energy ≤ reference + `energy` Ha **and** satisfying `criteria` (`singles` or `doubles`) |
| `"emin emax,criteria"` (e.g. `"0.1 0.5,singles"`) | Same as above, but restricted to an MF energy **band** `[reference + emin, reference + emax]` |
| `"energy,doubles_linked_singles"` | Singles within the energy window, plus doubles that factorize into allowed singles |

In `"0.5,singles"`, the boson expansion includes the reference determinant and all single excitations whose mean-field energy lies within **0.5 Ha** of the reference. Swaps among degenerate orbitals are not counted as excitations.

> **Note:** Determinant filtering is applied when the molecule has **point-group symmetry enabled** in PySCF (`mol.symmetry = True`) and a CI checkpoint is provided. Without symmetry, pass an explicit determinant list or use `tol` on the CI coefficients instead.

#### `use_symm` — symmetry selection rules

When `use_symm=True` and the PySCF molecule carries a supported point group, pyqmc builds a **symmetry mask** (`det_prod_filter`) over determinant pairs. Entry `(l, n)` is `True` only if the total irrep of determinants `l` and `n` match, so the overlap ⟨l\|n⟩ and related matrix elements can be nonzero.

This mask is applied in the ABVMC and ABCDMC matrix accumulators (`ab_vmc_excitations`, `abc_dmc_excitations`) to skip symmetry-forbidden blocks, reducing cost and avoiding spurious contributions. Set `use_symm=False` to accumulate the full determinant × determinant matrix (e.g. for debugging or when symmetry is not set up in PySCF).

Supported groups are those in pyqmc’s character table; unsupported groups (e.g. `Dooh`) are handled gracefully and symmetry masking is skipped.

## Project layout

```
pyqmc/
├── pyqmc/              # Core library
│   ├── recipes.py      # Standard OPTIMIZE / VMC / DMC recipes
│   ├── bosonrecipes.py # ABOPTIMIZE / ABVMC / ABDMC recipes
│   ├── mc.py, dmc.py   # VMC and DMC drivers
│   ├── bosonmc.py, bosondmc.py
│   └── ...
├── examples/
│   ├── boson/              # ABCDMC production examples (LDA, PBE Be⁺)
│   ├── he_recipe.py        # Standard Slater–Jastrow workflow
│   └── ...
├── tests/              # Unit and integration tests
└── doc/                # Sphinx documentation
```

## Documentation

Build the docs locally:

```bash
cd doc
pip install -r requirements.txt
make html
```

The built site is in `doc/build/html/`. Tutorials cover installation, wavefunction setup, and both recipe-level and low-level APIs.

Additional notes:

- [HPC installation](doc/source/specific_instructions.md)
- [Common problems](doc/source/common_problems.md)
- [Benchmarking](doc/source/benchmarking.md)

## Testing

```bash
pip install pytest "pyscf[naive_hci]"
pytest
```

## Citation

If you use the upstream pyqmc code, please cite the Wagner Group software and relevant publications from [WagnerGroup/pyqmc](https://github.com/WagnerGroup/pyqmc).

If you use ABCDMC methods from this fork, please cite the corresponding ABCDMC publication (add reference when available).

## License

MIT License — see [LICENSE](LICENSE). Original pyqmc copyright © Lucas K. Wagner.

## Acknowledgments

This project builds on [pyqmc](https://github.com/WagnerGroup/pyqmc) by the Wagner Group at the University of Illinois.

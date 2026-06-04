# HOWTO: Create a Conda Environment for the PersonLabCU SLEAP Repo

This guide is for PersonLabCU members who want to clone this repository, create a
fresh conda environment, and run this version of SLEAP.

The commands below install the project from this repository in editable mode, so
the `sleap` command uses the code in your local checkout.

## 1. Install prerequisites

Install these first if you do not already have them:

- Git: <https://git-scm.com/downloads>
- uv: <https://docs.astral.sh/uv/getting-started/installation/>
- Miniforge, Miniconda, or Anaconda
- Access to the private `PersonLabCU/sleap` GitHub repository

On Windows, use **Anaconda Prompt**, **Miniforge Prompt**, or a terminal where
`conda` is available.

Check that the tools are available:

```bash
git --version
conda --version
```

## 1a. Install uv (if needed)

If `uv --version` fails, install `uv` first.

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

On macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then close and reopen your terminal and verify:

```bash
uv --version
```

## 2. Clone the repository

Clone the PersonLabCU repository, not the upstream Talmo Lab repository:

```bash
git clone https://github.com/PersonLabCU/sleap.git
cd sleap
```

If you prefer SSH and have SSH keys configured with GitHub:

```bash
git clone git@github.com:PersonLabCU/sleap.git
cd sleap
```

## 3. Create a new conda environment

Use Python 3.13. This project supports Python `>=3.11,<3.14`, so do not use
Python 3.14.

```bash
conda create -y -n personlab-sleap -c conda-forge python=3.13 pip git
conda activate personlab-sleap
```

Upgrade the Python build tools in the new environment:

```bash
python -m pip install --upgrade pip setuptools==81.0.0 wheel
```

`setuptools` is pinned to `81.0.0` because current `torch` wheels used by this
project require `setuptools<82`.

## 4. Install the project and dependencies

Pick **one** of the install commands below.

### Option A: NVIDIA GPU, recommended for training

Use this on Windows or Linux workstations with an NVIDIA GPU. The PyTorch CUDA
wheels include the CUDA runtime libraries needed by PyTorch. You do not need to
install CUDA with conda for this environment, and your system CUDA toolkit is
not used by these PyTorch wheels.

```bash
python -m pip install -e ".[nn,anipose,jupyter]" --extra-index-url https://download.pytorch.org/whl/cu128
```

```bash
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

By default, use `cu128` (CUDA 12.8). If needed for a different machine, you can
switch backends by changing the extra index URL:

```bash
# NVIDIA GPU, CUDA 11.8 wheels
python -m pip install -e ".[nn,anipose,jupyter]" --extra-index-url https://download.pytorch.org/whl/cu118

# NVIDIA GPU, CUDA 13.0 wheels
python -m pip install -e ".[nn,anipose,jupyter]" --extra-index-url https://download.pytorch.org/whl/cu130
```

Choose the backend based on your NVIDIA driver compatibility on that machine
(not on whether a local CUDA toolkit is already installed).

### Option B: CPU only

Use this on machines without an NVIDIA GPU, on macOS, or when you only need the
GUI, labeling, proofreading, and lightweight testing.

```bash
python -m pip install -e ".[nn,anipose,jupyter]" --extra-index-url https://download.pytorch.org/whl/cpu
```

### Option C: Isolated uv project environment (safe side-by-side install)

Use this if users already have another SLEAP installation on their machine and
you want this repository to stay fully isolated. This creates a repo-local
`.venv` and does not replace the globally installed `sleap` tool.

```bash
# NVIDIA GPU (CUDA 12.8 default in this repo)
uv sync --extra nn --extra anipose --extra jupyter

# CPU-only
uv sync --extra nn-cpu --extra anipose --extra jupyter
```

For NVIDIA GPUs, you can also choose:

```bash
# CUDA 11.8
uv sync --extra nn-cuda118 --extra anipose --extra jupyter

# CUDA 13.0
uv sync --extra nn-cuda130 --extra anipose --extra jupyter
```

Run SLEAP from this isolated environment with:

```bash
uv run sleap
```

Avoid `uv tool install` for this fork if users also keep another `sleap` tool
installed globally, since both use the same command name.

These commands install:

- The core SLEAP GUI and command-line tools from this repository
- `sleap-io[all]` for `.slp`, video, analysis HDF5, and NWB file IO
- `sleap-nn[torch]` for training and inference
- `sleap-anipose` and OpenCV contrib support for multiview/3D projection tools
- Jupyter/JupyterLab for notebook workflows
- All base dependencies listed in `pyproject.toml`

## 5. Verify the environment

Run these checks from inside the activated environment:

```bash
python -m pip check
sleap doctor
python -c "import sleap, sleap_io, sleap_nn, aniposelib, h5py, PySide6; print('PersonLabCU SLEAP environment OK')"
```

If `sleap doctor` reports a GPU backend, the GPU install is ready for training.
If it reports CPU only, the GUI and CPU workflows should still work, but
training and inference will be much slower.

## 6. Run SLEAP

Start the GUI:

```bash
sleap
```

Or explicitly run the labeler:

```bash
sleap-label
```

Run command-line prediction after you have a trained model:

```bash
sleap-nn-track --help
```

## 7. Updating later

To update your local checkout to the latest PersonLabCU version:

```bash
conda activate personlab-sleap
git pull
python -m pip install -e ".[nn,anipose,jupyter]" --extra-index-url https://download.pytorch.org/whl/cu128
```

If your machine needs a different NVIDIA backend, replace `cu128` with
`cu118` or `cu130` in the index URL.

For a CPU-only environment, use the CPU install command instead:

```bash
python -m pip install -e ".[nn,anipose,jupyter]" --extra-index-url https://download.pytorch.org/whl/cpu
```

## Troubleshooting

### `conda activate` is not recognized

Open Anaconda Prompt or Miniforge Prompt and try again. If needed, initialize
conda for your shell:

```bash
conda init
```

Then close and reopen the terminal.

### GitHub asks for a username/password

GitHub does not accept account passwords for Git over HTTPS. Use a GitHub
personal access token when prompted, or configure SSH keys and clone with the
SSH URL.

### The GUI does not open because of Qt or OpenCV errors

Make sure you are in the clean conda environment:

```bash
conda activate personlab-sleap
where python
where sleap
```

On macOS/Linux, use:

```bash
which python
which sleap
```

Then rerun:

```bash
sleap doctor
```

### Recreate the environment from scratch

If the environment gets into a bad state, remove it and repeat the setup:

```bash
conda deactivate
conda env remove -n personlab-sleap
```


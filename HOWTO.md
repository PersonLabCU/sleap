# HOWTO: Install and Update PersonLabCU SLEAP

This guide is for PersonLabCU members who want to download and run the lab's
custom version of SLEAP. It assumes no prior experience with Git or Python.

The commands below install the project from this repository in editable mode, so
the `sleap` command uses the code in your local checkout.

The current lab version is on the `integrate-upstream-v1.6.4` branch. It combines
official SLEAP v1.6.4 with the PersonLabCU docked windows, multiview/3D tools,
reach-analysis tools, and other lab-specific GUI changes.

> **Important:** Do not install `sleap==1.6.4` from PyPI for this setup. That
> installs the official version without the PersonLabCU additions. Follow this
> guide so SLEAP runs from the PersonLabCU repository.

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

On Windows, first choose a folder that is **not inside OneDrive**. Keeping the
code and its Python environment outside OneDrive avoids file-locking and
permission errors during installation.

Open Anaconda Prompt, Miniforge Prompt, or Command Prompt and run:

```bash
cd /d C:\
mkdir repos
cd repos
git clone --branch integrate-upstream-v1.6.4 https://github.com/PersonLabCU/sleap.git
cd sleap
git branch --show-current
```

The last command should print:

```text
integrate-upstream-v1.6.4
```

What these commands do:

1. Create and enter `C:\repos`.
2. Download the PersonLabCU repository.
3. Select the branch containing official SLEAP v1.6.4 and the lab additions.
4. Enter the downloaded repository and confirm the selected branch.

On macOS or Linux, use:

```bash
mkdir -p ~/repos
cd ~/repos
git clone --branch integrate-upstream-v1.6.4 https://github.com/PersonLabCU/sleap.git
cd sleap
git branch --show-current
```

If you already have the PersonLabCU repository on your computer, do not clone a
second copy. Open a terminal in the existing repository and run:

```bash
git status
git fetch origin
git switch integrate-upstream-v1.6.4
git pull --ff-only
```

If `git switch` says the branch does not exist, run this once:

```bash
git switch --track origin/integrate-upstream-v1.6.4
```

Before switching branches or pulling updates, `git status` should not show
uncommitted work that you need to keep. Ask a lab member for help if it does.

## 3. Create a new conda environment

Follow this section if you will use Option A or Option B below. Skip it if you
will use the isolated `uv` environment in Option C.

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

### Option A: Conda with NVIDIA GPU, recommended for training

Use this on Windows or Linux workstations with an NVIDIA GPU. The PyTorch CUDA
wheels include the CUDA runtime libraries needed by PyTorch. You do not need to
install CUDA with conda for this environment, and your system CUDA toolkit is
not used by these PyTorch wheels.

```bash
python -m pip install -e ".[nn,anipose,jupyter]" --extra-index-url https://download.pytorch.org/whl/cu128
```

```bash
python -m pip install --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
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

### Option B: Conda with CPU only

Use this on machines without an NVIDIA GPU, on macOS, or when you only need the
GUI, labeling, proofreading, and lightweight testing.

```bash
python -m pip install -e ".[nn,anipose,jupyter]" --extra-index-url https://download.pytorch.org/whl/cpu
```

### Option C: Isolated uv project environment (safe side-by-side install)

Use this if users already have another SLEAP installation on their machine and
you want this repository to stay fully isolated. This creates a repo-local
`.venv` and does not replace the globally installed `sleap` tool. Do not use
this option when the repository is stored inside OneDrive.

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
python -c "from importlib.metadata import version; print('sleap:', version('sleap')); print('sleap-io:', version('sleap-io')); print('sleap-nn:', version('sleap-nn'))"
git branch --show-current
```

For this release, the version output should show:

```text
sleap: 1.6.4
sleap-io: 0.9.2
sleap-nn: 0.3.1
```

The final command should show `integrate-upstream-v1.6.4`.

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

Updating has two parts:

1. `git pull` downloads the latest PersonLabCU code.
2. Reinstalling or syncing updates Python packages when their required versions
   change.

Open a terminal and enter the repository. If you followed the Windows example
above:

```bash
cd /d C:\repos\sleap
git status
git switch integrate-upstream-v1.6.4
git pull --ff-only
```

If `git status` reports changes you need to keep, stop and ask for help before
pulling.

Then update the environment using the same installation method you originally
selected.

For the Conda NVIDIA GPU environment:

```bash
conda activate personlab-sleap
python -m pip install -e ".[nn,anipose,jupyter]" --extra-index-url https://download.pytorch.org/whl/cu128
```

For the Conda CPU-only environment:

```bash
conda activate personlab-sleap
python -m pip install -e ".[nn,anipose,jupyter]" --extra-index-url https://download.pytorch.org/whl/cpu
```

For the isolated `uv` environment:

```bash
uv sync --extra nn --extra anipose --extra jupyter
uv run sleap
```

After a Conda update, start SLEAP with `sleap`. After a `uv` update, start it
with `uv run sleap`.

## 8. Publishing the integration branch (repository maintainers only)

Most lab members do not need this section. A maintainer publishes the branch to
the PersonLabCU GitHub repository with:

```bash
git switch integrate-upstream-v1.6.4
git push -u origin integrate-upstream-v1.6.4
```

After the branch has been tested, merge it into the lab's stable branch and
create an organization-specific tag such as `v1.6.4-personlab.1`. Do not reuse
the official `v1.6.4` tag.

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

### `uv sync` reports "Access is denied" inside `.venv`

This commonly happens when the repository and `.venv` are stored inside
OneDrive. First close SLEAP, Python, Jupyter, and any editor using the
environment. Then deactivate it:

```bash
deactivate
```

Delete only the repository's `.venv` folder in File Explorer. Do not delete the
repository itself. Move or re-clone the repository somewhere outside OneDrive,
such as `C:\repos\sleap`, and recreate the environment:

```bash
uv sync --extra nn --extra anipose --extra jupyter
uv run sleap
```

### Recreate the environment from scratch

If the environment gets into a bad state, remove it and repeat the setup:

```bash
conda deactivate
conda env remove -n personlab-sleap
```

# Day 1 – Session 1 Instructor Notes

**Duration**: 3 hours

## Overview

This first session focuses on:

1. **Introduction & Objectives**  
2. **Project and Environment Setup**  
3. **Git & GitHub Best Practices**  
4. **HPC Basics: Slurm & Cedar**  
5. **Q&A and Wrap-Up**

The goal is to establish a solid foundation for the rest of the workshop. By the end of Session 1, participants should be able to:
- Understand the main objectives of the workshop.
- Set up a consistent Python environment for deep learning (conda/venv, PyTorch, etc.).
- Demonstrate basic Git and GitHub workflows (branching, pull requests, versioning).
- Understand how to access and run jobs on Cedar or a similar HPC cluster via Slurm.

---

## 1. Introduction & Objectives (15–20 min)

### Key Talking Points

- **Workshop Roadmap**  
  - Overview of the 2-day schedule: what’s coming in Day 1 vs. Day 2.
  - Explain what we’ll build: a multi-task deep learning project on the CUB Birds dataset (classification + bounding box regression).  
  - Emphasize “best practices” mindset: reproducibility, experiment tracking, HPC usage, etc.

- **Participant Roles and Goals**  
  - (Optional) Quick poll: who has used PyTorch / HPC / Git before?
  - Encourage a collaborative spirit—everyone’s expected to code along, experiment, and ask questions.

#### Instructor Notes

- Keep it interactive. Ask participants about their experience with HPC or Git to gauge how deeply you need to introduce concepts.
- Mention that by the end of Day 2, they will have a reproducible pipeline that they can adapt for their own research.

---

## 2. Project and Environment Setup (30–45 min)

### Steps to Cover

1. Create a project directory using Cookiecutter.
2. Create a pip environment.
3. Install needed dependencies.
4. Create a GitHub repository.
5. Push the initial project structure to GitHub.

### Project Directory Structure

[Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) is a handy tool that automatically creates a project directory structure for you. It’s a great way to start a new data science project.

We can install it using `pip install cookiecutter-data-science
` and then run the command:

```bash
ccds
```

### Environment Setup

We will be using venv and pip for environment management. I personally like Conda, and the new tool people like is UV, but when working on slurm it's easier to use venv and pip.

**Create a virtual environment**:  

```bash
python3 -m venv dlworkshop
source dlworkshop/bin/activate
pip install --upgrade pip
```

**Install dependencies**:  

```bash
pip install torch torchvision pandas scikit-learn matplotlib lightning wandb optuna jupyter
pip freeze > requirements.txt
```

## 3. Git & GitHub Best Practices (30 - 45 min)
### Key Concepts
- **Version Control**: Why it’s important for reproducibility and collaboration.
- **Branching**: How to create branches for features or experiments.
- **Pull Requests**: How to review and merge code changes.
- **Commit Messages**: Best practices for writing clear and informative commit messages.
- **.gitignore**: How to set up a `.gitignore` file to avoid committing unnecessary files.

### GitHub Command Line Tool

- **Installation**: Make sure students have the GitHub CLI installed. You can check by running `gh --version`.
- **Login**: Students should log in to GitHub using the command line. This can be done with `gh auth login`. They will need to follow the prompts to authenticate.
- **Creating a Repository**: Use `gh repo create <repo-name>` to create a new repository directly from the command line.
- **Cloning a Repository**: Use `gh repo clone <repo-name>` to clone an existing repository.

### Getting Started with Git

1. Create a new GitHub repository

Go to the GitHub website and create a new repository. Name it something like `dl-workshop-2025`. It can be empty since we will push our local code to it.

2. Initialize Git in your project directory

```bash
git init
```

3. Set up a commit and push

```bash
git remote add origin <your-github-repo-url>
git add .
git commit -m "Initial commit"
git push -u origin main
```

### Downloading the Dataset

We will be using the CUB bird dataset for this workshop. You can download it from the following link:

[Images and Annotations](https://data.caltech.edu/records/20098)

```bash
mkdir data
cd data
wget https://data.caltech.edu/records/20098/files/CUB_200_2011.tgz
tar -xvzf CUB_200_2011.tgz
```

## 4. HPC Basics: Slurm & Cedar (DRAC) (45–60 min)

### Overview of DRAC Clusters
The Digital Research Alliance of Canada (DRAC), formerly Compute Canada, provides several HPC clusters (Cedar, Graham, Béluga, Narval, etc.) designed to handle general-purpose and large parallel computing tasks. These clusters offer a mix of CPU, GPU, and large-memory nodes to support a variety of workloads, including deep learning.

### Module Loading
DRAC uses software modules to manage environments. Modules simplify loading specific software versions and dependencies.

- List available modules:
  ```bash
  module avail
  ```

- Find details about a module:
  ```bash
  module spider python
  ```

- Load a module (example):
  ```bash
  module load StdEnv/2023 intel/2023.2.1 cuda/11.8 python/3.10.13
  ```

Modules should be loaded at the start of your Slurm scripts or interactive sessions.

### Requesting GPUs
To request GPUs, use the following syntax in your Slurm job script:

```bash
#SBATCH --gpus-per-node=v100:1
```

- Available GPU types include `p100`, `v100`, `t4`, `a100`. Specifying GPU type ensures appropriate hardware allocation.
- Check GPU status and usage:

```bash
nvidia-smi
```

### Data Storage on DRAC Clusters

DRAC provides three primary storage types:

| Storage   | Purpose                       | Backup  | Quota       | Purge Policy?     |
|-----------|-------------------------------|---------|-------------|-------------------|
| `/home`   | User files, configs, scripts  | Yes     | Small fixed | No                |
| `/scratch`| High-performance, temporary   | No      | Large fixed | Yes (inactive data purged) |
| `/project`| Persistent research data      | Yes     | Adjustable  | No                |

- For performance, stage large datasets into `$SLURM_TMPDIR` at job start.

Example:

```bash
cp /home/user/scratch/dataset.h5 $SLURM_TMPDIR/
```

Always copy important output back to `/project` after computation.

### Advanced Slurm Configuration

A robust Slurm script might look like this:

```bash
#!/bin/bash
#SBATCH --job-name=my_deep_learning_job
#SBATCH --account=def-myaccount
#SBATCH --time=7-00:00:00
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-user=youremail@example.com
#SBATCH --mail-type=ALL

# Load modules
module load StdEnv/2023 intel/2023.2.1 cuda/11.8 python/3.10.13

# Move to job directory
cd $SLURM_TMPDIR

# Copy dataset to local disk
cp /home/user/scratch/full_dataset.h5 $SLURM_TMPDIR/

# Create virtual environment
virtualenv --no-download env
source env/bin/activate

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index -r /home/user/projects/my_project/requirements.txt

# Run script
srun python /home/user/projects/my_project/train.py \
    --data $SLURM_TMPDIR/full_dataset.h5 \
    --batch_size 32 \
    --epochs 100

# Copy checkpoints back to project directory
cp -r $SLURM_TMPDIR/checkpoints /home/user/projects/my_project/
```

### Python Virtual Environments

Always use a Python virtual environment for reproducibility and to manage package installations:

- Create environment:

```bash
module load python/3.10
virtualenv --no-download myenv
source myenv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
```

- Avoid creating virtual environments permanently on `/scratch` as they can be purged. Use `/project` or create temporarily within jobs as shown above.

### Transferring Data

For efficient data transfer, use:

- **rsync**: for quick synchronization and incremental transfers:

```bash
rsync -avz /source/directory user@cedar.computecanada.ca:/destination/directory
```

- **Globus**: Web-based for large datasets, high reliability.

### Job Monitoring

- Check job status:

```bash
squeue -u $USER
```

- Detailed job info:

```bash
scontrol show job <job_id>
```

### Interactive Session

For testing and debugging, request interactive resources:

```bash
srun --ntasks=1 --cpus-per-task=4 --mem=16G --time=0-00:30 --gpus-per-node=1 --pty bash
```

### Best Practices

- Test configurations interactively before batch submission.
- Regularly check resource usage with `sacct` and quotas with `quota`.
- Carefully manage your data storage to avoid exceeding quotas or losing data.


## 5. Q&A and Wrap-Up (15–20 min)

If there's time, we can discuss some of the following:

- **IDEs**
  - Jupyter Notebooks
  - PyCharm
  - VSCode
  - DataSpell / PyCharm

- **Python Management**
  - Conda vs. venv
  - Pip vs. Poetry
  - UV

- **Linters and Formatters**
  - Black
  - Flake8
  - isort
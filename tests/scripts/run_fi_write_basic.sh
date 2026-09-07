#!/bin/bash
#SBATCH --job-name=fi-write-basic
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=parry
#SBATCH --time=00:05:00
#SBATCH --account=datascience_collab

echo "=== fi_write basic test === $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
srun --mpi=cray_shasta $HOME/mori-ofi/build-ofi/tests/fi_write_basic_test
echo "=== Done $(date) ==="

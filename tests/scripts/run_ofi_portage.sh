#!/bin/bash
#SBATCH --job-name=mori-ofi-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --partition=parry
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

BUILD=${HOME}/mori-ofi/build-ofi
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu4 < /dev/urandom | tr -d ' ')
export MORI_IO_LOG_LEVEL=debug

echo "=== MORI OFI Slingshot test === $(date)"
echo "Nodes: $SLURM_NODELIST"
srun --ntasks=2 --nodes=2 --ntasks-per-node=1 --mpi=cray_shasta ${BUILD}/tests/ofi_slingshot_test
echo "=== Done $(date) ==="

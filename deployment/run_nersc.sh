#!/bin/bash
#SBATCH -A m1234
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3
#SBATCH --job-name=my_simulation
#SBATCH --output=slurm.out

export SLURM_CPU_BIND="cores"

# Make sure environment is set
source env_mpi_nersc.sh

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl="self,tcp"
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export OMPI_MCA_opal_cuda_support=true
export PMIX_MCA_psec=native

which mpiexec
echo 'tasks:' $SLURM_NTASKS

srun --mpi=pmi2 -n $SLURM_NTASKS w_run --work-manager mpi --n-workers=64

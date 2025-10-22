# Environment setup script (env_mpi.sh)
# >>> mamba initialize >>>
# !! Contents within this block are managed by 'micromamba shell init' !!
export MAMBA_EXE='/u/jdoe/bin/micromamba';
export MAMBA_ROOT_PREFIX='/u/jdoe/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
# <<< mamba initialize <

micromamba activate myenv
export HDF5_USE_FILE_LOCKING=0 # processes have trouble writing to hdf5 otherwise
export MPI=1
export UCX_TLS=tcp
export UCX_NET_DEVICES=eth1
export WEST_SIM_ROOT="$PWD"
export SIM_NAME=$(basename $WEST_SIM_ROOT)
export WM_N_WORKERS=1

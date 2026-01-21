#!/bin/bash

# ==========================================
# 1. Environment Configuration
# ==========================================

# [Network Settings]
# The subnet for the Thunderbolt Bridge (or your specific interface).
NETWORK_IF="10.0.0.0/24"

# [Host: Local Machine]
HOST_NAME="localhost"
HOST_SLOTS="1"                                      # Number of processes to run locally
HOST_PYTHON="/path/to/your/local/venv/bin/python"   # Path to local Python executable
HOST_DIR="/path/to/your/local/project"              # Local working directory

# [Node: Remote Machine]
NODE_IP="10.0.0.2"                                  # IP address of the remote node
NODE_USER="remote_username"                         # Remote SSH username
NODE_SLOTS="1"                                      # Number of processes to run remotely
NODE_PYTHON="/path/to/remote/venv/bin/python"       # Path to remote Python executable
NODE_DIR="/path/to/remote/project"                  # Remote working directory

# ==========================================
# 2. Execution Logic
# ==========================================

SCRIPT_NAME=$1
shift # Remove the first argument (script name) to pass the rest to python

if [ -z "$SCRIPT_NAME" ]; then
  echo "Usage: ./run_cluster.sh [python_script.py] [args...]"
  exit 1
fi

echo "==================================================="
echo " Launching Cluster Job..."
echo " - Host: $HOST_NAME ($HOST_SLOTS process)"
echo " - Node: $NODE_USER@$NODE_IP ($NODE_SLOTS process)"
echo " - WorkDir: $HOST_DIR"
echo "==================================================="

# Execute MPI command
# Note: 'oob_tcp_if_include' is critical for forcing Thunderbolt Bridge communication.
mpirun \
    --mca btl_tcp_if_include $NETWORK_IF \
    --mca oob_tcp_if_include $NETWORK_IF \
    -tag-output \
    -x DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH \
    -x PATH \
    -x OMPI_MCA_btl_tcp_if_include=$NETWORK_IF \
    -x OMPI_MCA_oob_tcp_if_include=$NETWORK_IF \
    -np $HOST_SLOTS -H $HOST_NAME:$HOST_SLOTS -wdir $HOST_DIR \
    $HOST_PYTHON $SCRIPT_NAME "$@" \
    : \
    -np $NODE_SLOTS -H $NODE_USER@$NODE_IP:$NODE_SLOTS -wdir $NODE_DIR \
    $NODE_PYTHON $SCRIPT_NAME "$@"
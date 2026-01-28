PROJ_ROOT=$(pwd)
BIN_NAME="core1_titan_engine"
LOG_DIR="$PROJ_ROOT/logs"
mkdir -p $LOG_DIR
 
export NCCL_DEBUG=WARN                     
export NCCL_P2P_ENABLE=1                   
export NCCL_IB_DISABLE=0                   
export CUDA_DEVICE_MAX_CONNECTIONS=1       
export OMP_NUM_THREADS=16                  

echo "🛠️ Verifying Build Status..."
if [ ! -f "$BIN_NAME" ]; then
    echo "⚠️ Binary not found. Running Make..."
    make all
fi

NUM_GPUS=$(nvidia-smi -L | wc -l)
BATCH_SIZE_PER_GPU=64
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE_PER_GPU))
DATASET_PATH="$PROJ_ROOT/data/frontier_v1.bin"

echo "------------------------------------------------"
echo "🚀 Launching Core 1 Training Cluster"
echo "🔥 Target Hardware: $NUM_GPUS x NVIDIA A100"
echo "📦 Global Batch Size: $TOTAL_BATCH_SIZE"
echo "------------------------------------------------"

stdbuf -oL ./$BIN_NAME \
    --dataset $DATASET_PATH \
    --batch $BATCH_SIZE_PER_GPU \
    --gpus $NUM_GPUS \
    --mode "ultra_power" \
    2>&1 | tee $LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "✅ Core 1 training cycle completed successfully."
else
    echo "❌ Training crashed! Check logs in $LOG_DIR for details."
fi

#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs) # e.g. 50
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --learning-rate) # e.g. 0.01
      LEARNING_RATE="$2"
      shift # past argument
      shift # past value
      ;;
    --data-directory) # e.g. 0.2
      DATA_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    --out-directory) # e.g. (96,96,3)
      OUT_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z "$EPOCHS" ]; then
    echo "Missing --epochs"
    exit 1
fi
if [ -z "$LEARNING_RATE" ]; then
    echo "Missing --learning-rate"
    exit 1
fi
if [ -z "$DATA_DIRECTORY" ]; then
    echo "Missing --data-directory"
    exit 1
fi
if [ -z "$OUT_DIRECTORY" ]; then
    echo "Missing --out-directory"
    exit 1
fi

OUT_DIRECTORY=$(realpath $OUT_DIRECTORY)
DATA_DIRECTORY=$(realpath $DATA_DIRECTORY)

# convert Edge Impulse dataset (in Numpy format, with JSON for labels into something YOLOv3 understands)
python3 -u extract_dataset.py --data-directory $DATA_DIRECTORY --out-directory /tmp/data

# Disable W&B prompts
export WANDB_MODE=disabled


cd /app/Edgelab

# add the current directory to the PYTHONPATH
export PYTHONPATH=/app/Edgelab:$PYTHONPATH

python3 -u tools/train.py mmpose configs/pfld/pfld_mv2n_112.py \
    --cfg-options load_from=/app/pfld_mv2n_112.pth \
    data.samples_per_gpu=8 \
    data.workers_per_gpu=1 \
    lr_config.warmup_iters=20 \
    optimizer.lr=$LEARNING_RATE \
    total_epochs=$EPOCHS \
    --data=/tmp/data 

echo "Training complete"
echo ""

mkdir -p $OUT_DIRECTORY

# copy the model to the output directory
cp /app/Edgelab/work_dirs/pfld_mv2n_112/exp1/latest.pth $OUT_DIRECTORY/model.pth

python3 ./tools/export.py /app/Edgelab/work_dirs/pfld_mv2n_112/exp1/pfld_mv2n_112.py  --type fp32 --weights $OUT_DIRECTORY/model.pth  --shape 112

mv $OUT_DIRECTORY/model_fp32.tflite $OUT_DIRECTORY/model.tflite

python3 ./tools/export.py /app/Edgelab/work_dirs/pfld_mv2n_112/exp1/pfld_mv2n_112.py  --type int8 --weights $OUT_DIRECTORY/model.pth --data /tmp/data/train/images --shape 112

mv $OUT_DIRECTORY/model_int8.tflite $OUT_DIRECTORY/model_quantized_int8_io.tflite
 
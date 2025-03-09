CONFIG_FILE=$1
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export ENCODER_PATH="/purestorage/project/tyk/project24/models/encoders"
export FACE_ENCODER_PATH="/purestorage/project/tyk/project24/models/model_ir_se50.pth"
export OUTPUT_DIR="/purestorage/project/tyk/project24/logs"
export CACHE_DIR="/purestorage/project/tyk/tmp"


accelerate launch --config_file $CONFIG_FILE /purestorage/project/tyk/project24/ArtfaceStudio_ML/train/ip-adapter/face_emb_train.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --image_encoder_path=$ENCODER_PATH \
    --mixed_precision="fp16" \
    --resolution=512 \
    --train_batch_size=8 \
    --dataloader_num_workers=4 \
    --learning_rate=1e-04 \
    --save_steps=10000 \
    --output_dir=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR \
    --face_encoder_path=$FACE_ENCODER_PATH
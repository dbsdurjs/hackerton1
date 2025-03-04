export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export BASE_INSTANCE_DIR="./data_dir/"
export CLASS_DIR="./class_dir"
export BASE_OUTPUT_DIR="./saved_results_exp"

unique_identifiers=("sks" "zwx" "swz" "kql" "ifg" "wpb")

for i in $(seq 1 6)
do

  INSTANCE_DIR="${BASE_INSTANCE_DIR}train_soldier${i}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/result_soldier${i}"

  echo "Processing ${INSTANCE_DIR} and saving to ${OUTPUT_DIR}"
  echo "instance prompt : a photo of camouflaged ${unique_identifiers[$i-1]} soldier"

  python train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=${INSTANCE_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --class_data_dir=$CLASS_DIR \
    --with_prior_preservation --prior_loss_weight=0.1 \
    --mixed_precision="fp16" \
    --instance_prompt="a photo of camouflaged ${unique_identifiers[$i-1]} soldier" \
    --class_prompt="a realistic photo of camouflaged soldier in hiding" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=800 \
    --push_to_hub \
    
done

DEVICE=1 # gpu num

RANDOM_ID="VQA-RS"
RESULTS_FILE="results_${RANDOM_ID}.json"

torchrun open_flamingo/eval/evaluate.py \
    --lm_path "/data/share/pyz/llm_deploy/checkpoints/llama-7b" \
    --lm_tokenizer_path "/data/share/pyz/llm_deploy/checkpoints/llama-7b" \
    --checkpoint_path "/data/share/OpenFlamingo/openflamingo_checkpoint.pt" \
    --vision_encoder_path "ViT-L-14" \
    --vision_encoder_pretrained 'openai' \
    --device $DEVICE \
    --vqav2_train_image_dir_path "/data/wyl/coco_data/train2014"  \
    --vqav2_train_questions_json_path "/data/pyz/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
    --vqav2_train_annotations_json_path  "/data/pyz/data/vqav2/v2_mscoco_train2014_annotations.json" \
    --vqav2_test_image_dir_path "/data/wyl/coco_data/val2014/" \
    --vqav2_test_questions_json_path "/data/share/pyz/data/vqav2/v2_mscoco_val2014_question_subdata.json" \
    --vqav2_test_annotations_json_path "/data/share/pyz/data/vqav2/v2_mscoco_val2014_annotations_subdata.json" \
    --results_file $RESULTS_FILE \
    --num_samples 5000 \
    --shots 0 4 \
    --num_trials 1 \
    --seed 5 \
    --batch_size 8 \
    --cross_attn_every_n_layers 4 \
    --precision fp16 \
    --eval_vqav2
    
echo "evaluation complete! results written to $RESULTS_FILE"

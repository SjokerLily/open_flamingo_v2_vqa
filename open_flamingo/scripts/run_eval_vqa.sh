DEVICE=7 # gpu num

RESULTS_FILE="openflamingo_v2_vqaV2_RS_SIIR_SQQR.json"

export MASTER_ADDR='localhost'
export MASTER_PORT='10086'

python /data/ll/StyleCaption/open_flamingo_v2_os/open_flamingo/eval/evaluate.py \
    --lm_path "/data/share/mpt-7b/" \
    --lm_tokenizer_path "/data/share/mpt-7b/" \
    --checkpoint_path "/data/share/OpenFlamingo-9B-vitl-mpt7b/models--openflamingo--OpenFlamingo-9B-vitl-mpt7b/snapshots/e6e175603712c7007fe3b9c0d50bdcfbd83adfc2/checkpoint.pt" \
    --vision_encoder_path "ViT-L-14" \
    --vision_encoder_pretrained 'openai' \
    --device $DEVICE \
    --vqav2_train_image_dir_path "/data/wyl/coco_data/train2014"  \
    --vqav2_train_questions_json_path "/data/pyz/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
    --vqav2_train_annotations_json_path  "/data/pyz/data/vqav2/v2_mscoco_train2014_annotations.json" \
    --vqav2_test_image_dir_path "/data/wyl/coco_data/val2014/" \
    --vqav2_test_questions_json_path "/data/share/pyz/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json" \
    --vqav2_test_annotations_json_path "/data/share/pyz/data/vqav2/v2_mscoco_val2014_annotations.json" \
    --results_file $RESULTS_FILE \
    --num_trials 1 \
    --seed 5 \
    --batch_size 1 \
    --cross_attn_every_n_layers 4 \
    --precision fp16 \
    --eval_vqav2 \

echo "evaluation complete! results written to $RESULTS_FILE"

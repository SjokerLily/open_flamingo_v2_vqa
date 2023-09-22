from huggingface_hub import hf_hub_download
import torch

# save_path = "/data/share/OpenFlamingo-9B-vitl-mpt7b"
# checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b",
#                                   "checkpoint.pt", cache_dir=save_path)
# print(checkpoint_path)

save_path = "/data/share/mpt-7b"
import transformers
# model = transformers.AutoModelForCausalLM.from_pretrained(
#   'mosaicml/mpt-7b',
#   trust_remote_code=True
# )
print(transformers.__version__)
model = transformers.AutoTokenizer.from_pretrained(
        'mosaicml/mpt-7b', local_files_only=True,
        trust_remote_code=True,
    )
model.save_pretrained(save_directory=save_path, trust_remote_code=True, revision="main")

# from open_flamingo_v2.open_flamingo import create_model_and_transforms
#
# model, image_processor, tokenizer = create_model_and_transforms(
#     clip_vision_encoder_path="ViT-L-14",
#     clip_vision_encoder_pretrained="openai",
#     lang_encoder_path="/data/share/pyz/llm_deploy/checkpoints/llama-7b",
#     tokenizer_path="/data/share/pyz/llm_deploy/checkpoints/llama-7b",
#     cross_attn_every_n_layers=4
# )
#
# model.load_state_dict(torch.load(checkpoint_path), strict=False)

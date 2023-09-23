import argparse
import copy
import importlib
import json
import os
import random
import uuid
from collections import defaultdict
import sys
sys.path.append("/data/ll/StyleCaption")  # where open_flamingo_v2_os is

from einops import repeat
import more_itertools
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import (
    CaptionDataset,
    VQADataset,
    ImageNetDataset,
    HatefulMemesDataset,
)
from tqdm import tqdm


from eval_datasets import VQADataset, ImageNetDataset
from classification_utils import (
    IMAGENET_CLASSNAMES,
    IMAGENET_1K_CLASS_ID_TO_LABEL,
    HM_CLASSNAMES,
    HM_CLASS_ID_TO_LABEL,
)

from eval_model import BaseEvalModel

from ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo_v2_os.open_flamingo.src.flamingo import Flamingo
from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation

from open_flamingo_v2_os.open_flamingo.train.distributed import init_distributed_device, world_info_from_env

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)
# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
)

parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument(
    "--no_caching_for_classification",
    action="store_true",
    help="Use key-value caching for classification evals to speed it up. Currently this doesn't underperforms for MPT models.",
)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_coco",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)
parser.add_argument(
    "--eval_vqav2",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQAV2.",
)
parser.add_argument(
    "--eval_ok_vqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on OK-VQA.",
)
parser.add_argument(
    "--eval_vizwiz",
    action="store_true",
    default=False,
    help="Whether to evaluate on VizWiz.",
)
parser.add_argument(
    "--eval_textvqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on TextVQA.",
)
parser.add_argument(
    "--eval_imagenet",
    action="store_true",
    default=False,
    help="Whether to evaluate on ImageNet.",
)
parser.add_argument(
    "--eval_flickr30",
    action="store_true",
    default=False,
    help="Whether to evaluate on Flickr30.",
)
parser.add_argument(
    "--eval_hateful_memes",
    action="store_true",
    default=False,
    help="Whether to evaluate on Hateful Memes.",
)

# Dataset arguments

## Flickr30 Dataset
parser.add_argument(
    "--flickr_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--flickr_karpathy_json_path",
    type=str,
    help="Path to the dataset_flickr30k.json file.",
    default=None,
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
)
## COCO Dataset
parser.add_argument(
    "--coco_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_val_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_karpathy_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default=None,
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_retrieval_result_dir_path",
    type=str,
    default=None,
)

## OK-VQA Dataset
parser.add_argument(
    "--ok_vqa_train_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_image_dir_path",
    type=str,
    help="Path to the vqav2/val2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_val2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_val2014_annotations.json file.",
    default=None,
)

## VizWiz Dataset
parser.add_argument(
    "--vizwiz_train_image_dir_path",
    type=str,
    help="Path to the vizwiz train images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_image_dir_path",
    type=str,
    help="Path to the vizwiz test images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)

# TextVQA Dataset
parser.add_argument(
    "--textvqa_image_dir_path",
    type=str,
    help="Path to the textvqa images directory.",
    default=None,
)
parser.add_argument(
    "--textvqa_train_questions_json_path",
    type=str,
    help="Path to the textvqa questions json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_train_annotations_json_path",
    type=str,
    help="Path to the textvqa annotations json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_test_questions_json_path",
    type=str,
    help="Path to the textvqa questions json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_test_annotations_json_path",
    type=str,
    help="Path to the textvqa annotations json file.",
    default=None,
)

## Imagenet dataset
parser.add_argument("--imagenet_root", type=str, default="/tmp")

## Hateful Memes dataset
parser.add_argument(
    "--hateful_memes_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--hateful_memes_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--hateful_memes_test_annotations_json_path",
    type=str,
    default=None,
)

# Distributed evaluation
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)


def main():
    args, leftovers = parser.parse_known_args()
    # open_flamingo_v2_os.open_flamingo.eval.
    module = importlib.import_module(f"models.{args.model}")

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args, eval_model.device)
    # device_id = "cuda:" + str(model_args["device"])
    print(device_id)
    # device_id = torch.device(device_id)
    eval_model.set_device(device_id)
    eval_model.init_distributed()

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    if args.eval_ok_vqa:
        print("Evaluating on OK-VQA...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                ok_vqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="ok_vqa",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} OK-VQA score: {ok_vqa_score}")
                    scores.append(ok_vqa_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean OK-VQA score: {np.nanmean(scores)}")
                results["ok_vqa"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_vqav2:
        print("Evaluating on VQAv2...")
        retrieval_type_set = ["RS", "SIIR", "SQQR"]
        for retrieval_type in retrieval_type_set:
            for shot in args.shots:
                scores = []
                print("retrieval type: ", retrieval_type)
                print("shot: ", shot)
                for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                    vqa_score = evaluate_vqa(
                        args=args,
                        eval_model=eval_model,
                        num_shots=shot,
                        seed=seed,
                        dataset_name="vqav2",
                        retrieval_type=retrieval_type,
                    )
                    if args.rank == 0:
                        print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                        scores.append(vqa_score)

                if args.rank == 0:
                    print(f"Shots {shot} Mean VQA score: {np.nanmean(scores)}")
                    results["vqav2"].append(
                        {"retrieval_type": retrieval_type,
                         "shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                    )

    if args.eval_vizwiz:
        print("Evaluating on VizWiz...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vizwiz_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vizwiz",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} VizWiz score: {vizwiz_score}")
                    scores.append(vizwiz_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean VizWiz score: {np.nanmean(scores)}")
                results["vizwiz"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )


    if args.rank == 0 and args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + query_set_size must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def get_query_set(train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, batch_size, seed):
    np.random.seed(seed)
    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )
    return loader


def sample_batch_demos_from_query_set(query_set, num_samples, batch,
                                      retrieval_type, clip, seed, query_set_size,
                                      control_signals):
    if not clip:
        query_set = get_query_set(query_set, query_set_size, seed)
        return [random.sample(query_set, num_samples) for _ in range(len(batch["image"]))]
    else:
        return [[query_set.id2item(id) for id in batch[retrieval_type][i][:num_samples]] for i in range(len(batch["image"]))]


def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def evaluate_vqa(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "vqav2",
    retrieval_type = None,
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: accuracy score
    """

    if dataset_name == "ok_vqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path
    elif dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
        retrieval_path = args.vqav2_retrieval_result_dir_path
    elif dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path
    elif dataset_name == "textvqa":
        train_image_dir_path = args.textvqa_image_dir_path
        train_questions_json_path = args.textvqa_train_questions_json_path
        train_annotations_json_path = args.textvqa_train_annotations_json_path
        test_image_dir_path = args.textvqa_image_dir_path
        test_questions_json_path = args.textvqa_test_questions_json_path
        test_annotations_json_path = args.textvqa_test_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # train_dataset = VQADataset(
    #     image_dir_path=train_image_dir_path,
    #     question_path=train_questions_json_path,
    #     annotations_path=train_annotations_json_path,
    #     is_train=True,
    #     dataset_name=dataset_name,
    # )
    #
    # test_dataset = VQADataset(
    #     image_dir_path=test_image_dir_path,
    #     question_path=test_questions_json_path,
    #     annotations_path=test_annotations_json_path,
    #     is_train=False,
    #     dataset_name=dataset_name,
    # )

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    predictions = []

    if retrieval_type == "RS":
        control_signals = {"clip": False,
                           "retrieval_type": retrieval_type,
                           }
        random_uuid = "RS_{}".format(num_shots)
    else:
        control_signals = {"clip": True,
                       "retrieval_type": retrieval_type,
                       }
        random_uuid = "{}_{}".format(control_signals["retrieval_type"],
                                     num_shots,
                                     )

    print("control signals of prompt: ", control_signals)

    train_dataset = VQADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=dataset_name,
    )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
        retrieval_path=retrieval_path,
    )

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    def get_vqa_prompt(question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    np.random.seed(
        seed + args.rank
    )  # make sure each worker has a different seed for the random context samples
    if not os.path.exists(f"{dataset_name}results_{random_uuid}.json"):
        for batch in tqdm(
            test_dataloader,
            desc=f"Running inference {dataset_name}",
            disable=args.rank != 0,
        ):
            batch_demo_samples = sample_batch_demos_from_query_set(
                train_dataset, effective_num_shots, batch,
                control_signals["retrieval_type"], control_signals["clip"],
                seed, args.query_set_size,
                control_signals=control_signals)

            batch_images = []
            batch_text = []
            for i in range(len(batch["image"])):
                in_context_samples = batch_demo_samples[i]

                if num_shots > 0:
                    context_images = [x["image"] for x in in_context_samples]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])

                context_text = "".join(
                    [
                        get_vqa_prompt(
                            question=x["question"], answer=x["answers"][0]
                        )
                        for x in in_context_samples
                    ]
                )

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                batch_text.append(
                    context_text + get_vqa_prompt(question=batch["question"][i])
                )

            try:
                outputs = eval_model.get_outputs(
                    batch_images=batch_images,
                    batch_text=batch_text,
                    min_generation_length=min_generation_length,
                    max_generation_length=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                )
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception


            process_function = (
                postprocess_ok_vqa_generation
                if dataset_name == "ok_vqa"
                else postprocess_vqa_generation
            )

            new_predictions = map(process_function, outputs)

            # for new_prediction, sample_id in zip(new_predictions, batch["question_id"]):
            #     predictions.append({"answer": new_prediction, "question_id": sample_id})
            new_predictions = list(new_predictions)
            for i in range(len(batch["image"])):
                predictions.append({"answer": new_predictions[i], "question_id": batch["question_id"][i],
                                    "prompt_text": batch_text[i],
                                    "prompt_images": [img['image_id'] for img in batch_demo_samples[i]],
                                    "question": batch["question"][i]
                                    })

        # all gather
        all_predictions = [None] * args.world_size
        torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists
        if args.rank != 0:
            return

        all_predictions = [
            item for sublist in all_predictions for item in sublist
        ]  # flatten

        # save the predictions to a temporary file
        # random_uuid = str(uuid.uuid4())
        with open(f"{dataset_name}results_{random_uuid}.json", "w") as f:
            f.write(json.dumps(all_predictions, indent=4))

    if test_annotations_json_path is not None:
        acc, _ = compute_vqa_accuracy(
            f"{dataset_name}results_{random_uuid}.json",
            test_questions_json_path,
            test_annotations_json_path,
        )
        # delete the temporary file
        # os.remove(f"{dataset_name}results_{random_uuid}.json")

    else:
        print("No annotations provided, skipping accuracy computation.")
        print("Temporary file saved to:", f"{dataset_name}results_{random_uuid}.json")
        acc = None

    return acc



if __name__ == "__main__":
    main()

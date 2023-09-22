import json
import os
from vqa_metric import compute_vqa_accuracy

num_shots = 4
control_signals = {"clip": False,
                   "retrieval_type": "clip_images",
                   "specification": True,
                   "order": "order"}

if control_signals["clip"]:
    random_uuid = "{}_{}_spec{}_{}".format(control_signals["retrieval_type"],
                                           num_shots,
                                           control_signals["specification"],
                                           control_signals["order"])
else:
    random_uuid = "rs_{}_spec{}_{}".format(
        num_shots,
        control_signals["specification"],
        control_signals["order"])

dataset_name = "vqav2"
test_questions_json_path = "/data/share/pyz/data/vqav2/v2_mscoco_val2014_question_subdata.json"
test_annotations_json_path = "/data/share/pyz/data/vqav2/v2_mscoco_val2014_annotations_subdata.json"

print("Evaluating json file:")
# print(f"{dataset_name}results_{random_uuid}.json")
path = "/data/ll/StyleCaption/open_flamingo_v2/open_flamingo/eval/vqav2results_SQAQAR_16_specFalse_order.json"
print(path)

acc, evalResult = compute_vqa_accuracy(
            path,
            test_questions_json_path,
            test_annotations_json_path,
        )

print("The accuracy of VQA is : ", acc)

result_path = "vqa_evalAcc/"
if not os.path.exists(result_path):
    os.makedirs(result_path)
with open(result_path + f"{dataset_name}results_{random_uuid}_{acc}.json", "w") as f:
    f.write(json.dumps(evalResult, indent=4))

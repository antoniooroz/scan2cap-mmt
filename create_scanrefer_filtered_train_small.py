import os
import json
import random

def get_scene_list(scanrefer_train):
    scene_set = set()
    for item in scanrefer_train:
        scene_set.add(item["scene_id"])
    return list(scene_set)

def get_filtered_train(scanrefer_train, scene_list):
    scanrefer_eval_train = []
    for item in scanrefer_train:
        if item["scene_id"] in scene_list:
            scanrefer_eval_train.append(item)
    return scanrefer_eval_train

scanrefer_train = json.load(open(os.path.join("./", "ScanRefer_filtered_train.json")))

scene_list = get_scene_list(scanrefer_train)
print("Num scenes original: " + str(len(scene_list)))

fraction_of_training_data = 0.1 # Fraction of training data used for evaluation during training

scene_list_filtered = random.choices(scene_list, k=int(fraction_of_training_data * len(scene_list)))

print("Num scenes filtered: " + str(len(scene_list_filtered)))

scanrefer_eval_train = get_filtered_train(scanrefer_train, scene_list_filtered)

with open('ScanRefer_filtered_train_small.json', 'w', encoding='utf-8') as f:
    json.dump(scanrefer_eval_train, f, ensure_ascii=False, indent=4)
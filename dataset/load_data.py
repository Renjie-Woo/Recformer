# 预处理数据集
from utils import read_json
import os
import json
import random


# train/val/test: {"user":[item_id,], }
# meta: {"asin": {item_info}}
# smap: {"asin": item_id}

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as wf:
        json.dump(data, wf)


def get_train_data(seq):
    seq_len = len(seq)
    start = min(0, seq_len)
    pos = random.randint(start, seq_len - 1)
    return seq[:pos], [seq[pos]]


def load_data(dir):
    train_path = os.path.join(dir, "train.json")
    val_path = os.path.join(dir, "val.json")
    test_path = os.path.join(dir, "test.json")
    meta_path = os.path.join(dir, "meta_data.json")
    smap_path = os.path.join(dir, "smap.json")

    train_data = read_json(train_path, False)
    val_data = read_json(val_path, False)
    test_data = read_json(test_path, False)
    meta_data = read_json(meta_path, False)
    smap_data = read_json(smap_path, False)

    # print(len(train_data))
    # print(len(val_data))
    # print(len(test_data))

    train_keys = sorted(train_data.keys(), key=lambda x: int(x))
    val_keys = sorted(val_data.keys(), key=lambda x: int(x))
    test_keys = sorted(test_data.keys(), key=lambda x: int(x))
    # print(train_keys[0], train_keys[-1])
    # print(val_keys[0], val_keys[-1])
    # print(test_keys[0], test_keys[-1])
    #
    # demo = train_keys - val_keys
    # print(demo)

    meta_data_with_item_id = {}
    for asin in smap_data:
        item_id = smap_data[asin]
        item_info = meta_data[asin]
        meta_data_with_item_id[item_id] = item_info

    train_final = {}
    train_final_label = {}
    for user_id in train_keys:
        seq = train_data[user_id]
        seq_data = parse_seq(seq, meta_data_with_item_id)
        seq_input, seq_label = get_train_data(seq_data)
        train_final[user_id] = {
            'input': seq_data
        }
        train_final_label[user_id] = {
            "input": seq_input,
            "label": seq_label
        }

    val_final = {}
    for user_id in val_keys:
        train_seq = train_data[user_id]
        val_seq = val_data[user_id]
        val_final[user_id] = {
            "input": parse_seq(train_seq, meta_data_with_item_id),
            "label": parse_seq(val_seq, meta_data_with_item_id)
        }

    test_final = {}
    for user_id in test_keys:
        train_seq = train_data[user_id]
        test_seq = test_data[user_id]
        test_final[user_id] = {
            "input": parse_seq(train_seq, meta_data_with_item_id),
            "label": parse_seq(test_seq, meta_data_with_item_id)
        }

    save_json(train_final, "./Arts/train.json")
    save_json(val_final, "./Arts/val.json")
    save_json(test_final, "./Arts/test.json")
    save_json(train_final_label, "./Arts/train_with_label.json")


def parse_seq(item_id_seq, meta_map):
    item_seq = []
    for item_id in item_id_seq:
        item_seq.append(meta_map[item_id])
    return item_seq


if __name__ == '__main__':
    dir = "./finetune_data_dataset/Arts"
    load_data(dir)

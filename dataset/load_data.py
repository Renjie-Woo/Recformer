from utils import read_json

# train/val/test: {"user":[item_id,], }
    # meta: {"asin": {item_info}}
    # smap: {"asin": item_id}

def load_data():
    train_path = ""
    val_path = ""
    test_path = ""
    meta_path = ""
    smap_path = ""

    train_data = read_json(train_path, False)
    val_data = read_json(val_path, False)
    test_data = read_json(test_path, False)
    meta_data = read_json(meta_path, False)
    smap_data = read_json(smap_path, False)




from utils import read_json
import os
from dataset.item_dataset import ItemDataset
from torch.utils.data import DataLoader
from recformer_v2.tokenizer import RecformerTokenizer
from recformer_v2.model import RecformerModel

train_path = "train_with_label.json"
val_path = "val.json"
test_path = "test.json"

tokenizer = RecformerTokenizer.from_pretrained("allenai/longformer-base-4096")
print(tokenizer)

def load_data(dir):
    train_data = read_json(os.path.join(dir, train_path), True)
    val_data = read_json(os.path.join(dir, val_path), True)
    test_data = read_json(os.path.join(dir, test_path), True)

    return train_data, val_data, test_data


def generate_dataloader(dir):
    train_data, val_data, test_data = load_data(dir)
    train_dataset = ItemDataset(train_data, tokenizer)
    val_dataset = ItemDataset(val_data, tokenizer)
    test_dataset = ItemDataset(test_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=test_dataset.collate_fn)
    return train_loader, val_loader, test_loader

    


if __name__ == '__main__':
    model = RecformerModel()
    train_loader, _, _ = generate_dataloader('./dataset/Arts')
    for inputs, labels in train_loader:
        logits = model(**inputs)
        print(logits)
        break
        
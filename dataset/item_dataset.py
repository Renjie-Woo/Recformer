from torch.utils.data import Dataset


class ItemDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.len = len(data)

    def __getitem__(self, index):
        item = self.data[index]
        return item

    def __len__(self):
        return self.len

    def collate_fn(self, batch_data):
        inputs = [data['input'] for data in batch_data]
        labels = [data['label'] for data in batch_data]
        tokenized_inputs = self.tokenizer(inputs, return_tensor=True)
        tokenized_labels = self.tokenizer(labels, return_tensor=True)
        return tokenized_inputs, tokenized_labels


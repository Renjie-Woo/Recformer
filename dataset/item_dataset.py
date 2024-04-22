from torch.utils.data import Dataset, DataLoader
class ItemDataset(Dataset):

    def __init__(self, interactions, meta, id2item):
        interactions_list = interactions.items()
        interactions_list = sorted(interactions_list, key=lambda x: x[0])



    def __getitem__(self, item):

    def __len__(self):
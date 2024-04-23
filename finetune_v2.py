from utils import read_json, AverageMeterSet, Ranker
import os
from dataset.item_dataset import ItemDataset
from torch.utils.data import DataLoader
from recformer_v2.tokenizer import RecformerTokenizer
from recformer_v2.model import RecformerModel, RecformerForSeqRec
from optimization import create_optimizer_and_scheduler
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

train_path = "train_v1.json"
val_path = "val_v1.json"
test_path = "test_v1.json"
meta_path = "meta.json"

device = torch.device('cuda:1')#torch.device('cuda:{}'.format('0')) if 1>=0 else torch.device('cpu')
print(device)
fp16 = True
verbose = 3
gradient_accumulation_steps = 8
scaler=scaler = torch.cuda.amp.GradScaler()
num_train_epochs = 16

tokenizer = RecformerTokenizer.from_pretrained("allenai/longformer-base-4096")
tokenizer.add_device(device)
#print(tokenizer)

def load_data(dir):
    train_data = read_json(os.path.join(dir, train_path), True)
    val_data = read_json(os.path.join(dir, val_path), True)
    test_data = read_json(os.path.join(dir, test_path), True)
    meta_data = read_json(os.path.join(dir, meta_path), True)
    return train_data, val_data, test_data, meta_data


def generate_dataloader(dir):
    train_data, val_data, test_data, meta_data = load_data(dir)
    train_dataset = ItemDataset(train_data, tokenizer)
    val_dataset = ItemDataset(val_data, tokenizer)
    test_dataset = ItemDataset(test_data, tokenizer)
    meta_dataset = ItemDataset(meta_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=test_dataset.collate_fn)
    meta_loader = DataLoader(meta_dataset, batch_size=16, shuffle=False, collate_fn=meta_dataset.collate_fn)
    return train_loader, val_loader, test_loader, meta_loader

def encode_all_items(model, dataloader):
    model.eval()

    item_embeddings = []
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(tqdm(dataloader)):  
            outputs = model(**inputs)
            item_embeddings.append(outputs.pooler_output.detach())

    item_embeddings = torch.cat(item_embeddings, dim=0)#.cpu()

    return item_embeddings

def eval(model, dataloader):

    model.eval()

    ranker = Ranker([10,50])
    average_meter_set = AverageMeterSet()

    for batch, labels in tqdm(dataloader, ncols=100, desc='Evaluate'):

        # for k, v in batch.items():
        #     batch[k] = v
        # labels = labels

        with torch.no_grad():
            scores = model(**batch)

        res = ranker(scores, labels)

        metrics = {}
        for i, k in enumerate([10, 50]):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()

    return average_metrics


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler):

    model.train()

    for step, (inputs, labels) in enumerate(tqdm(dataloader, ncols=100, desc='Training')):
        

        if fp16:
            with autocast():
                loss = model(**inputs)
        else:
            loss = model(**inputs)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        if fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if fp16:

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                optimizer.zero_grad()

                if optimizer_was_run:
                    scheduler.step()

            else:

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                optimizer.zero_grad()

if __name__ == '__main__':
    train_loader, val_loader, test_loader, meta_loader = generate_dataloader('./dataset/Arts')

    #pretrain_ckpt = ""

    model = RecformerForSeqRec()
    model = model.to(device)
    #pretrain_ckpt = torch.load(pretrain_ckpt)
    #model.load_state_dict(pretrain_ckpt, strict=False)

    for param in model.longformer.embeddings.word_embeddings.parameters():
        param.requires_grad = False

    # path_item_embeddings = dir_preprocess / f'item_embeddings_{path_corpus.name}'
    # if path_item_embeddings.exists():
    #     print(f'[Item Embeddings] Use cache: {path_tokenized_items}')
    # else:
    #     print(f'Encoding items.')
    #     item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
    #     torch.save(item_embeddings, path_item_embeddings)
    #
    # item_embeddings = torch.load(path_item_embeddings)
    #item_embeddings = encode_all_items(model.longformer, meta_loader)
    #torch.save(item_embeddings, './item_embeddings.pt')
    item_embeddings = torch.load('./item_embeddings.pt')
    print('load item embeddings from local')
    model.init_item_embedding(item_embeddings)

    num_train_optimization_steps = int(len(train_loader) / gradient_accumulation_steps) * num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps)
    
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    test_metrics = eval(model, test_loader)
    print(f'Test set: {test_metrics}')

    best_target = float('-inf')
    patient = 5

    path_ckpt = "./model.pt"

    for epoch in range(500):

        item_embeddings = encode_all_items(model.longformer, meta_loader)
        model.init_item_embedding(item_embeddings)

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
    
        if (epoch + 1) % verbose == 0:
            dev_metrics = eval(model, val_loader)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')
    
            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 5
                torch.save(model.state_dict(), path_ckpt)
    
            else:
                patient -= 1
                if patient == 0:
                    break
    
    print('Load best model in stage 1.')
    model.load_state_dict(torch.load(path_ckpt))
    
    patient = 3
    
    for epoch in range(num_train_epochs):
    
        train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
    
        if (epoch + 1) % verbose == 0:
            dev_metrics = eval(model, val_loader)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')
    
            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 3
                torch.save(model.state_dict(), path_ckpt)
    
            else:
                patient -= 1
                if patient == 0:
                    break
    
    print('Test with the best checkpoint.')
    model.load_state_dict(torch.load(path_ckpt))
    test_metrics = eval(model, test_loader)
    print(f'Test set: {test_metrics}')
    


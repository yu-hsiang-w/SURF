import os
import json
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import transformers
from transformers import BertModel
import normalize_text


class Financial_Encoder(BertModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, add_pooling_layer=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        emb = nn.functional.normalize(emb, dim=-1)

        return emb


class InBatch(nn.Module):
    def __init__(self, retriever, tokenizer):
        super(InBatch, self).__init__()

        self.tokenizer = tokenizer
        self.encoder = retriever

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, **kwargs):
        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask)

        scores = torch.einsum("id, jd->ij", qemb / 0.05, kemb)
        loss = nn.functional.cross_entropy(scores, labels)

        predicted_idx = torch.argmax(scores, dim=-1)

        return loss


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup))

        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )


def set_optim(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)

    scheduler_args = {
        "warmup": 664,
        "total": 13281,
        "ratio": 0.0,
    }
    scheduler_class = WarmupLinearScheduler
    scheduler = scheduler_class(optimizer, **scheduler_args)
    return optimizer, scheduler


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        training=False,
        maxload=None,
        normalize=False,
    ):
        self.normalize_fn = normalize_text.normalize
        self.training = training
        self._load_data(datapaths, maxload)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question"]
        gold = example["positive_ctxs"][0]

        gold = gold["title"] + " " + gold["text"] if "title" in gold and len(gold["title"]) > 0 else gold["text"]

        example = {
            "query": self.normalize_fn(question),
            "gold": self.normalize_fn(gold),
        }
        return example

    def _load_data(self, datapaths, maxload):
        counter = 0
        self.data = []
        files = os.listdir(datapaths)
        for path in files:
            path = str(path)
            file_data, counter = self._load_data_json(datapaths, path, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, datapaths, path, counter, maxload=None):
        examples = []

        path = os.path.join(datapaths, path)

        with open(path, "r") as fin:
            data = json.load(fin)

        for example in data:
            counter += 1
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter


class Collator(object):
    def __init__(self, tokenizer, passage_maxlength=256):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        golds = [ex["gold"] for ex in batch]

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        kout = self.tokenizer.batch_encode_plus(
            golds,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
        }

        return batch

def finetuning(model, optimizer, scheduler, tokenizer):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    current_directory = os.getcwd()

    train_data = os.path.join(current_directory, "Training Data")
    train_dataset = Dataset(
        datapaths=train_data,
        normalize=False,
        maxload=None,
        training=True
    )

    valid_data = os.path.join(current_directory, "Validation Data")
    valid_dataset = Dataset(
        datapaths=valid_data,
        normalize=False,
        maxload=None,
        training=True
    )

    collator = Collator(tokenizer, passage_maxlength=256)

    train_sampler = RandomSampler(train_dataset)
    valid_sampler = RandomSampler(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=64,
        drop_last=True,
        num_workers=3,
        collate_fn=collator
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=64,
        drop_last=True,
        num_workers=3,
        collate_fn=collator
    )

    best_val_loss = float('inf')
    best_model_weights = None
    epochs_no_improve = 0

    epoch = 0
    while epoch < 50:
        epoch += 1
        model.train()

        for i, batch in enumerate(train_dataloader):
            print(f"{i}, {epoch}")
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            train_loss = model(**batch, stats_prefix="train")
            train_loss.backward()

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

        # Validation step
        val_loss_epoch = 0.0
        model.eval()

        with torch.no_grad():
            for batch in valid_dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                val_loss = model(**batch)
                val_loss_epoch += val_loss.item()
        
        val_loss_epoch /= len(valid_dataloader)
        
        # Check for improvement
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if (epochs_no_improve >= 3):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load the best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    model_save_path = "/home/yhwang/Fin_relation/Trained_Model"
    model.encoder.save_pretrained(model_save_path)
    print("Finish")


def main():

    torch.manual_seed(2024)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    retriever = Financial_Encoder.from_pretrained("bert-base-uncased")
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    
    model = InBatch(retriever, tokenizer)
    model = model.to(device)
    
    optimizer, scheduler = set_optim(model)
    
    finetuning(model, optimizer, scheduler, tokenizer)


if __name__ == "__main__":
    main()

import warnings
# Global warning configurations
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import random
from model import Code_Note
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import sklearn.exceptions
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
setSeed = False


class FocalLoss(nn.Module):
    """
    Standard Focal Loss implementation for imbalanced datasets.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (float, int)):
            self.register_buffer('alpha', torch.tensor([1 - alpha, alpha]))
        elif isinstance(alpha, (list, torch.Tensor)):
            self.register_buffer('alpha', torch.tensor(alpha))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            loss = self.alpha[targets] * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LALoss(nn.Module):
    """
    Logit Adjustment Loss implementation.
    """

    def __init__(self, num_classes=2, reduction='mean'):
        super(LALoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        batch_size = targets.size(0)
        class_counts = torch.bincount(targets, minlength=self.num_classes)
        weights = (batch_size - class_counts.float()) / batch_size
        weights = weights[targets]

        loss = F.cross_entropy(inputs, targets, reduction='none')
        weighted_loss = weights * loss
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


def setup_and_log_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000) % 1000000

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open("random_seeds.log", "a") as f:
        f.write(f"Seed: {seed}, Time: {time.ctime()}\n")

    print(f"Using random seed: {seed}")
    return seed


class Example(object):
    def __init__(self, code, text, label):
        self.code = code
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, inputs_code_ids, inputs_code_masks, inputs_text_ids, inputs_text_masks, label):
        self.inputs_code_ids = inputs_code_ids
        self.inputs_code_masks = inputs_code_masks
        self.inputs_text_ids = inputs_text_ids
        self.inputs_text_masks = inputs_text_masks
        self.label = label


def read_file(codefile, textfile):
    examples = []
    code_data = pd.read_csv(codefile, na_filter=False, encoding_errors='ignore')
    text_data = pd.read_csv(textfile, na_filter=False, encoding_errors='ignore')
    code = code_data['text'].values.tolist()
    code_label = code_data['label'].values.tolist()
    text = text_data['text'].values.tolist()
    text_label = text_data['label'].values.tolist()
    for c, cl, t, tl in zip(code, code_label, text, text_label):
        if c != '' and t != '' and int(cl) == int(tl):
            examples.append(Example(c, t, int(cl)))
        else:
            break
    return examples


def mini_sample(examples, num):
    example1 = []
    unique_numbers = random.sample(range(0, len(examples)), num)
    for n in unique_numbers:
        for example_index, example in enumerate(examples):
            if example_index == n:
                example1.append(example)
    return example1


def text_to_feature(examples, code_tokenizer, text_tokenizer, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # Tokenize code snippet
        raw_code_tokens = code_tokenizer.tokenize(example.code)[:510]
        cls_t = code_tokenizer.cls_token if code_tokenizer.cls_token else "<s>"
        sep_t = code_tokenizer.sep_token if code_tokenizer.sep_token else "</s>"

        code_tokens = [cls_t] + raw_code_tokens + [sep_t]
        inputs_code_ids = code_tokenizer.convert_tokens_to_ids(code_tokens)
        inputs_code_ids = [i if i is not None else code_tokenizer.unk_token_id for i in inputs_code_ids]
        inputs_code_masks = [1] * len(inputs_code_ids)

        code_padding_length = 512 - len(inputs_code_ids)
        inputs_code_ids += [code_tokenizer.pad_token_id] * code_padding_length
        inputs_code_masks += [0] * code_padding_length

        # Tokenize graph/text context
        raw_text_tokens = text_tokenizer.tokenize(example.text)[:510]
        cls_t_text = text_tokenizer.cls_token if text_tokenizer.cls_token else "<s>"
        sep_t_text = text_tokenizer.sep_token if text_tokenizer.sep_token else "</s>"

        text_tokens = [cls_t_text] + raw_text_tokens + [sep_t_text]
        inputs_text_ids = text_tokenizer.convert_tokens_to_ids(text_tokens)
        inputs_text_ids = [i if i is not None else text_tokenizer.unk_token_id for i in inputs_text_ids]
        inputs_text_masks = [1] * len(inputs_text_ids)

        text_padding_length = 512 - len(inputs_text_ids)
        inputs_text_ids += [text_tokenizer.pad_token_id] * text_padding_length
        inputs_text_masks += [0] * text_padding_length

        if example_index < 5 and stage == 'train':
            logger.info(f"*** Example {example_index} ***")

        features.append(
            InputFeatures(
                inputs_code_ids,
                inputs_code_masks,
                inputs_text_ids,
                inputs_text_masks,
                example.label
            )
        )
    return features


def evaluate(eval_dataloader, model, device):
    start_time = time.time()
    total_correct = 0.0
    total_examples = 0.0
    all_pre = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in eval_dataloader:
            inputs_code_id, inputs_code_mask, inputs_text_id, inputs_text_mask, inputs_label = [b.to(device) for b in
                                                                                                batch]
            mlp_output = model(inputs_code_id, inputs_code_mask, inputs_text_id, inputs_text_mask)
            pred = torch.argmax(mlp_output, dim=1)

            all_labels += inputs_label.tolist()
            all_pre += pred.tolist()
            total_correct += torch.sum(pred == inputs_label).item()
            total_examples += int(mlp_output.size(0))

    acc = total_correct / total_examples
    f1 = f1_score(y_true=all_labels, y_pred=all_pre, zero_division=0)
    rec = recall_score(y_true=all_labels, y_pred=all_pre, zero_division=0)
    prec = precision_score(y_true=all_labels, y_pred=all_pre, zero_division=0)

    execution_time = time.time() - start_time
    return {
        'acc': acc,
        'f1': f1,
        'rec': rec,
        'prec': prec,
        'execution_time': execution_time
    }


def main():
    epochs = 20
    batchsize = 4

    setSeed = True
    setup_and_log_seed()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    code_model_path = './codesage-small-v2'
    text_model_path = './roberta-base'

    code_tokenizer = AutoTokenizer.from_pretrained(code_model_path, trust_remote_code=True, local_files_only=True)
    code_model = AutoModel.from_pretrained(code_model_path, trust_remote_code=True, local_files_only=True)

    print(f"Code Model Hidden Size: {code_model.config.hidden_size}")

    text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
    text_model = RobertaModel.from_pretrained(text_model_path)

    code_hidden = code_model.config.hidden_size
    text_hidden = text_model.config.hidden_size

    model = Code_Note(code_model, text_model, code_size=code_hidden, text_size=text_hidden, hidden_size=1536, output_size=384)
    model.to(device)

    train_codefile = '../dataset/trvd/trvd_slice_train.csv'
    train_textfile = '../dataset/trvd/interp_train.csv'
    eval_codefile = '../dataset/trvd/trvd_slice_val.csv'
    eval_textfile = '../dataset/trvd/interp_val.csv'

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)

    # Load and process training dataset
    examples = read_file(train_codefile, train_textfile)
    train_examples = text_to_feature(examples, code_tokenizer, text_tokenizer, 'train')
    all_inputs_code_ids = torch.tensor([f.inputs_code_ids for f in train_examples])
    all_inputs_code_masks = torch.tensor([f.inputs_code_masks for f in train_examples])
    all_inputs_text_ids = torch.tensor([f.inputs_text_ids for f in train_examples])
    all_inputs_text_masks = torch.tensor([f.inputs_text_masks for f in train_examples])
    all_inputs_labels = torch.tensor([f.label for f in train_examples])
    train_data = TensorDataset(all_inputs_code_ids, all_inputs_code_masks, all_inputs_text_ids, all_inputs_text_masks,
                               all_inputs_labels)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True)

    # Load and process validation dataset
    eva_examples = read_file(eval_codefile, eval_textfile)
    eval_examples = text_to_feature(eva_examples, code_tokenizer, text_tokenizer, 'eval')
    all_evalinputs_code_ids = torch.tensor([f.inputs_code_ids for f in eval_examples])
    all_evalinputs_code_masks = torch.tensor([f.inputs_code_masks for f in eval_examples])
    all_evalinputs_text_ids = torch.tensor([f.inputs_text_ids for f in eval_examples])
    all_evalinputs_text_masks = torch.tensor([f.inputs_text_masks for f in eval_examples])
    all_evalinputs_labels = torch.tensor([f.label for f in eval_examples])
    eval_data = TensorDataset(all_evalinputs_code_ids, all_evalinputs_code_masks, all_evalinputs_text_ids,
                              all_evalinputs_text_masks, all_evalinputs_labels)
    eval_dataloader = DataLoader(eval_data, batch_size=batchsize, shuffle=True)

    best_metrics = {}
    best_epoch = 0
    with open("result.txt", "a") as f:
        for epoch in range(epochs):
            train_total_lose = 0.0
            train_total_correct = 0.0
            train_total_examples = 0.0

            # Training phase
            model.train()
            loop = tqdm((train_dataloader), total=len(train_dataloader))
            for bidx, batch in enumerate(loop):
                inputs_code_id, inputs_code_mask, inputs_text_id, inputs_text_mask, inputs_label = batch[0].to(device), \
                    batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
                optimizer.zero_grad()
                mlp_output = model(inputs_code_id, inputs_code_mask, inputs_text_id, inputs_text_mask)

                loss = criterion(mlp_output, inputs_label)
                loss.backward()
                optimizer.step()

                pred = torch.argmax(mlp_output, dim=1)
                train_total_lose += loss.item()
                correct = torch.sum(pred == inputs_label)
                train_total_correct += correct.item()
                train_total_examples += int(mlp_output.size(0))

                loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                loop.set_postfix({'Train Loss': f'{train_total_lose / len(train_dataloader)}',
                                  'Train ACC': f'{train_total_correct / train_total_examples}'})
            loop.close()

            # Validation phase
            metrics = evaluate(eval_dataloader, model, device)
            eval_acc = metrics['acc']
            eval_f1 = metrics['f1']
            eval_rec = metrics['rec']
            eval_prec = metrics['prec']
            eval_time = metrics['execution_time']

            f.write(f'Epoch [{epoch + 1}/{epochs}] val_time: {eval_time}seconds    val_acc={eval_acc}, val_f1={eval_f1}, val_recall={eval_rec}, val_precision={eval_prec}' + '\n')
            print(f'val_time: {eval_time}seconds    val_acc={eval_acc}, val_f1={eval_f1}, val_recall={eval_rec}, val_precision={eval_prec}')

            if epoch == 0:
                best_metrics = metrics
                best_epoch = epoch
            else:
                if eval_acc + eval_f1 >= best_metrics['acc'] + best_metrics['f1']:
                    best_metrics = metrics
                    best_epoch = epoch

            best_acc = best_metrics['acc']
            best_f1 = best_metrics['f1']
            best_rec = best_metrics['rec']
            best_prec = best_metrics['prec']

            print('---best epoch---')
            print(f'best epoch: {best_epoch + 1}    acc={best_acc}, f1={best_f1}, recall={best_rec}, precision={best_prec}')

        f.write(f'best epoch: {best_epoch + 1}    acc={best_acc}, f1={best_f1}, recall={best_rec}, precision={best_prec}' + '\n')

    if setSeed:
        with open("random_seeds.log", "a") as s:
            s.write(f'best epoch: {best_epoch + 1}    acc={best_acc}, f1={best_f1}, recall={best_rec}, precision={best_prec}' + '\n')


if __name__ == "__main__":
    main()
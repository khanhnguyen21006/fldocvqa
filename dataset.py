import os, json, random, logging
import numpy as np
from PIL import Image
from pathlib import Path
logging.getLogger('PIL').setLevel(logging.WARNING) # Fix the glitch of unknown logs from PIL

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from transformers import T5Tokenizer

from utils import *

class MixtureDocVQA(Dataset):
    def __init__(self, data_dir, split, tokenizer, indices=None, **kwargs):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = kwargs["transform"]
        self.hparam = kwargs
        data_list = []
        for _dset in kwargs["dataset"]:
            data = np.load(os.path.join(self.data_dir, _dset, split, "vqa.npy"), allow_pickle=True)
            if indices is not None:
                data = data[indices[_dset]]
            data_list.append(data)
        self.data = np.concatenate(data_list)

    def __getitem__(self, i):
        record = self.data[i]

        dataset = record["metadata"]["dataset"]
        question_id = record["metadata"].get("question_id", i)
        question = record["question"].lower()
        answers = [_ans.lower() for _ans in record['answers']]
        if len(record['ocr_tokens']) == 0:
            ocr_tokens = []
            ocr_boxes = np.empty([0, 4])
        else:
            ocr_tokens = [_token.lower() for _token in record['ocr_tokens']]
            ocr_boxes = np.array([_bbox for _bbox in record['ocr_normalized_boxes']])
        image_name = record['image_name']
        image = Image.open(os.path.join(self.data_dir, dataset, f"images/{image_name}.png")).convert("RGB")
        if self.transform:
            image = self.transform(image)

        ret = {
            'dataset': dataset,
            'question_id': question_id,
            'question': question,
            'answers': answers,
            'image_name': image_name,
            'image': image,
            'ocr_tokens': ocr_tokens,
            'ocr_boxes': ocr_boxes
        }
        return ret

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        keys = set([key for b in batch for key in b.keys()]) # List of dictionaries to dict of lists.
        dict_batch = {k: [b[k] if k in b else None for b in batch] for k in keys}

        dict_batch['image'] = torch.stack(dict_batch['image'])
        dict_batch['image_box'] = torch.stack([get_image_bbox(self.hparam["image_size"]) for _ in range(len(dict_batch['image']))])

        bs, max_ind = len(batch), 0
        max_source_len, max_target_len = self.hparam["max_source_len"], self.hparam["max_target_len"]
        prompt_box, eos_box = [0, 0, 0, 0], [0, 0, 0, 0]
        list_ids, list_boxes = [], []
        for _ind in range(bs):
            input_ids = self.tokenizer(dict_batch['question'][_ind]).input_ids[:-1]
            input_boxes = [prompt_box] * len(input_ids)

            for _token, _box in zip(dict_batch['ocr_tokens'][_ind], dict_batch['ocr_boxes'][_ind]):
                _token_ids = self.tokenizer(_token).input_ids[:-1]
                input_ids.extend(_token_ids)
                input_boxes.extend([_box]*len(_token_ids))

            list_ids.append(input_ids[:max_source_len-1] + [self.tokenizer.eos_token_id])
            list_boxes.append(np.concatenate([input_boxes[:max_source_len-1], np.array([eos_box])]))
            max_ind = min(max(max_ind, len(input_ids) + 1), max_source_len)

        batch_input_ids = torch.full([bs, max_ind], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        batch_input_boxes = torch.full([bs, max_ind, 4], fill_value=0, dtype=torch.float)
        batch_attention_masks = torch.zeros([bs, max_ind], dtype=torch.long)
        for _ind in range(bs):
            batch_input_ids[_ind, :len(list_ids[_ind])] = torch.LongTensor(list_ids[_ind])
            batch_input_boxes[_ind, :len(list_boxes[_ind])] = torch.from_numpy(list_boxes[_ind][:len(list_boxes[_ind])])
            batch_attention_masks[_ind, :len(list_ids[_ind])] = 1
        dict_batch['input_text_id'] = batch_input_ids
        dict_batch['input_text_mask'] = batch_attention_masks
        dict_batch['input_text_box'] = batch_input_boxes

        batch_labels = self.tokenizer(
            [random.choice(_ans) for _ans in dict_batch['answers']],
            return_tensors='pt', padding=True, truncation=True, max_length=max_target_len
        ).input_ids
        batch_labels[batch_labels[:] == self.tokenizer.pad_token_id] = -100
        dict_batch['label'] = batch_labels

        return dict_batch

class MixtureSSL(Dataset):
    def __init__(self, data_dir, split, tokenizer, indices=None, **kwargs):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = kwargs["transform"]
        self.hparam = kwargs
        data_list = []
        for _dset in kwargs["dataset"]:
            with open(os.path.join(self.data_dir, _dset, split, "ssl.json"), 'r') as f:
                data = np.array(json.load(f))
            if indices is not None:
                data = data[indices[_dset]]
            data_list.append(data)
        self.data = np.concatenate(data_list)

    def __getitem__(self, i):
        record = self.data[i]

        dataset = record["dataset"]
        if len(record['ocr_tokens']) == 0:
            ocr_tokens = []
            ocr_boxes = np.empty([0, 4])
        else:
            ocr_tokens = [_token.lower() for _token in record['ocr_tokens']]
            ocr_boxes = np.array([_bbox for _bbox in record['ocr_normalized_boxes']])
        image_name = record['document_id']
        image = Image.open(os.path.join(self.data_dir, dataset, f"images/{image_name}.png")).convert("RGB")
        if self.transform:
            image = self.transform(image)

        ret = {
            'dataset': dataset,
            'image_name': image_name,
            'image': image,
            'ocr_tokens': ocr_tokens,
            'ocr_boxes': ocr_boxes
        }
        return ret

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        keys = set([key for b in batch for key in b.keys()]) # List of dictionaries to dict of lists.
        dict_batch = {k: [b[k] if k in b else None for b in batch] for k in keys}

        dict_batch['image'] = torch.stack(dict_batch['image'])
        dict_batch['image_box'] = torch.stack([get_image_bbox(self.hparam["image_size"]) for _ in range(len(dict_batch['image']))])

        bs, max_source_len, max_target_len = len(batch), self.hparam["max_source_len"], self.hparam["max_target_len"]
        prompt_box, sent_box, eos_box = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]

        for _task in self.hparam["ssl_task"]:
            list_ids, list_boxes, list_labels = [], [], []
            max_input_ind, max_label_ind = 0, 0
            if _task == 'lm': # text: <extra_l_id_{}>text</extra_l_id_{}>, bbox:sent_box*(len(text)+2), prob:0.75, label:<extra_l_id_><100><350><118><372>
                for _ind in range(bs):
                    input_ids = self.tokenizer("layout modeling.").input_ids[:-1]
                    input_boxes = [prompt_box] * len(input_ids)
                    label = []

                    mask_count = 0
                    _selected = [_i for _i  in range(len(dict_batch['ocr_tokens'][_ind])) if random.random() < self.hparam["lm_prob"]]
                    for _i, (_token, _box) in enumerate(zip(dict_batch['ocr_tokens'][_ind], dict_batch['ocr_boxes'][_ind])):
                        if mask_count < 100 and _i in _selected:
                            _token_ids = [self.tokenizer._convert_token_to_id(f"<extra_l_id_{mask_count}>")]
                            _token_ids += self.tokenizer(_token).input_ids[:-1]
                            _token_ids += [self.tokenizer._convert_token_to_id(f"</extra_l_id_{mask_count}>")]
                            _token_boxes = [sent_box]*len(_token_ids)
                            if len(label) + 5 < max_target_len:
                                label.extend([
                                    self.tokenizer._convert_token_to_id(_item)
                                    for _item in ([f"<extra_l_id_{mask_count}>"] + [f"<loc_{round(_bc*500)}>" for _bc in _box])
                                ])
                            mask_count += 1
                        else:
                            _token_ids = self.tokenizer(_token).input_ids[:-1]
                            _token_boxes = [_box]*len(_token_ids)
                        input_ids.extend(_token_ids)
                        input_boxes.extend(_token_boxes)
                    assert len(label) < max_target_len and len(label) in [5*len(_selected), max_target_len-max_target_len%5]
                    list_ids.append(input_ids[:max_source_len-1] + [self.tokenizer.eos_token_id])
                    list_boxes.append(np.concatenate([input_boxes[:max_source_len-1], np.array([eos_box])]))
                    list_labels.append(label + [self.tokenizer.eos_token_id])
                    max_input_ind = min(max(max_input_ind, len(input_ids) + 1), max_source_len)
                    max_label_ind = min(max(max_label_ind, len(label) + 1), max_target_len)
            elif _task == 'tm': # text: <extra_t_id_{}><100><350><118><372></extra_t_id_{}>, bbox:sent_box*6, prob:0.75, label:<extra_t_id_{}>text
                tm_prob = 0.5
                for _ind in range(bs):
                    input_ids = self.tokenizer("text modeling").input_ids[:-1]
                    input_boxes = [prompt_box] * len(input_ids)
                    label = []

                    _selected = [_i for _i  in range(len(dict_batch['ocr_tokens'][_ind])) if random.random() < self.hparam["tm_prob"]]
                    mask_count = 0
                    for _i, (_token, _box) in enumerate(zip(dict_batch['ocr_tokens'][_ind], dict_batch['ocr_boxes'][_ind])):
                        if mask_count < 100 and _i in _selected:
                            _token_ids = [
                                self.tokenizer._convert_token_to_id(_item)
                                for _item in ([f"<extra_t_id_{mask_count}>"] + [f"<loc_{round(_bc*500)}>" for _bc in _box] + [f"</extra_t_id_{mask_count}>"])
                            ]
                            _token_boxes = [sent_box]*len(_token_ids)
                            _tmp_label = [self.tokenizer._convert_token_to_id(f"<extra_t_id_{mask_count}>")] + self.tokenizer(_token).input_ids[:-1]
                            if len(label) + len(_tmp_label) < max_target_len:
                                label.extend(_tmp_label)
                            mask_count += 1
                        else:
                            _token_ids = self.tokenizer(_token).input_ids[:-1]
                            _token_boxes = [_box]*len(_token_ids)
                        input_ids.extend(_token_ids)
                        input_boxes.extend(_token_boxes)
                    assert len(label) < max_target_len
                    list_ids.append(input_ids[:max_source_len-1] + [self.tokenizer.eos_token_id])
                    list_boxes.append(np.concatenate([input_boxes[:max_source_len-1], np.array([eos_box])]))
                    list_labels.append(label + [self.tokenizer.eos_token_id])
                    max_input_ind = min(max(max_input_ind, len(input_ids) + 1), max_source_len)
                    max_label_ind = min(max(max_label_ind, len(label) + 1), max_target_len)
            elif _task == 'tlm': # text: <extra_id_{}>, bbox:sent_box, prob:0.15, label:<extra_id_{}>text<100><350><118><372>
                tlm_prob = 0.15
                for _ind in range(bs):
                    input_ids = self.tokenizer("joint text-layout modeling.").input_ids[:-1]
                    input_boxes = [prompt_box] * len(input_ids)
                    label = []

                    mask_count = 0
                    _selected = [_i for _i  in range(len(dict_batch['ocr_tokens'][_ind])) if random.random() < self.hparam["tlm_prob"]]
                    for _i, (_token, _box) in enumerate(zip(dict_batch['ocr_tokens'][_ind], dict_batch['ocr_boxes'][_ind])):
                        if mask_count < 100 and _i in _selected:
                            _token_ids = [self.tokenizer._convert_token_to_id(f"<extra_id_{mask_count}>")]
                            _token_boxes = [sent_box]
                            _tmp_label = [self.tokenizer._convert_token_to_id(f"<extra_id_{mask_count}>")]
                            _tmp_label += self.tokenizer(_token).input_ids[:-1]
                            _tmp_label += [self.tokenizer._convert_token_to_id(f"<loc_{round(_bc*500)}>") for _bc in _box]
                            if len(label) + len(_tmp_label) < max_target_len:
                                label.extend(_tmp_label)
                            mask_count += 1
                        else:
                            _token_ids = self.tokenizer(_token).input_ids[:-1]
                            _token_boxes = [_box]*len(_token_ids)
                        input_ids.extend(_token_ids)
                        input_boxes.extend(_token_boxes)
                    assert len(label) < max_target_len
                    list_ids.append(input_ids[:max_source_len-1] + [self.tokenizer.eos_token_id])
                    list_boxes.append(np.concatenate([input_boxes[:max_source_len-1], np.array([eos_box])]))
                    list_labels.append(label + [self.tokenizer.eos_token_id])
                    max_input_ind = min(max(max_input_ind, len(input_ids) + 1), max_source_len)
                    max_label_ind = min(max(max_label_ind, len(label) + 1), max_target_len)
            else:
                raise ValueError(f"Invalid SSL task {_task}.")

            batch_input_ids = torch.full([bs, max_input_ind], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
            batch_input_boxes = torch.full([bs, max_input_ind, 4], fill_value=0, dtype=torch.float)
            batch_attention_masks = torch.zeros([bs, max_input_ind], dtype=torch.long)
            batch_labels = torch.full([bs, max_label_ind], fill_value=-100, dtype=torch.long) # add eos
            for _ind in range(bs):
                batch_input_ids[_ind, :len(list_ids[_ind])] = torch.LongTensor(list_ids[_ind])
                batch_input_boxes[_ind, :len(list_boxes[_ind])] = torch.from_numpy(list_boxes[_ind][:len(list_boxes[_ind])])
                batch_attention_masks[_ind, :len(list_ids[_ind])] = 1
                batch_labels[_ind, :len(list_labels[_ind])] = torch.LongTensor(list_labels[_ind])
            dict_batch[f'input_text_id_{_task}'] = batch_input_ids
            dict_batch[f'input_text_mask_{_task}'] = batch_attention_masks
            dict_batch[f'input_text_box_{_task}'] = batch_input_boxes
            dict_batch[f'label_{_task}'] = batch_labels

        return dict_batch

def get_dataloader(split, indices, batch_size, tokenizer, args):
    task, dset_cls = ("docvqa", MixtureDocVQA) if args.docvqa else ("ssl", MixtureSSL)
    dset_hparam = get_dataset_hparam(args)
    dset = dset_cls(args.data_dir, split, tokenizer, indices=indices, **dset_hparam)
    dloader = DataLoader(
        dataset=dset,
        batch_size=batch_size,
        shuffle=(split=="train"),
        collate_fn=dset.collate_fn,
        drop_last=(split=="train"),
        num_workers=args.num_worker,
        pin_memory=True,
    )
    return dloader, dset

def get_dataset_hparam(args):
    hparam = {
        "dataset": args.dataset.split(","),
        "transform": get_image_transform(),
        "image_size": args.image_size,
        "max_source_len": args.max_source_len,
        "max_target_len": args.max_target_len,
        "ssl_task": args.ssl_task.split(',') if args.ssl_task else None,
        "lm_prob": args.lm_prob if args.ssl_task and "lm" in args.ssl_task else 0,
        "tm_prob": args.tm_prob if args.ssl_task and "tm" in args.ssl_task else 0,
        "tlm_prob": args.tlm_prob if args.ssl_task and "tlm" in args.ssl_task else 0,
    }
    return hparam

import os, json, random, datetime, argparse
import logging

import editdistance
import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloader
from model import init_models
from utils import save_result

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Evaluator(metaclass=Singleton):
    def __init__(self, case_sensitive=False, ascending=True, **kwargs):
        self.case_sensitive = case_sensitive
        self.get_edit_distance = editdistance.eval
        self.anls_threshold = 0.5

        if kwargs.get("resume_from", None):
            self.load_state(os.path.join(kwargs["resume_from"]))
            assert self.curr_epoch < kwargs["num_round"]
        else:
            self.ascending = ascending
            self.curr_epoch = 0
            self.best_metric = -10000 if self.ascending else 10000
            self.best_epoch = 0

    def get_scores(self, gt_answers, preds, tasks):
        scores = []
        for _ind in range(len(preds)):
            task = tasks[_ind]
            gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[_ind]]
            pred = self._preprocess_str(preds[_ind])
            if task in ["docvqa", "infovqa"]:
                score = [self._calculate_anls(gt, pred), self._calculate_accuracy(gt, pred)]
                metric = ["anls", "accuracy"]
            elif task in "wtq":
                score = [self._calculate_anls(gt, pred)]
                metric = ["anls"]
            elif task in "tabfact":
                score = [int(any([gt_elm == pred for gt_elm in gt]))]
                metric = ["accuracy"]
            else:
                raise ValueError(f"Invalid task {task}.")
            scores.append({"task": task, "score": score, "metric": metric})
        return scores

    def update_global_metrics(self, curr_value, curr_epoch):
        self.curr_epoch += 1
        if self.ascending:
            if curr_value > self.best_metric:
                self.best_metric = curr_value
                self.best_epoch = curr_epoch
                return True
            return False
        else:
            if curr_value < self.best_metric:
                self.best_metric = curr_value
                self.best_epoch = curr_epoch
                return True
            return False

    def _preprocess_str(self, string):
        if not self.case_sensitive:
            string = string.lower()
        return string.strip()

    def _calculate_accuracy(self, gt, pred, answer_type='string'):
        if answer_type == 'not-answerable':
            return 1 if pred in ['', 'none', 'NA', None, []] else 0
        if pred == 'none' and answer_type != 'not-answerable':
            return 0
        for gt_elm in gt:
            if gt_elm == pred:
                return 1
        return 0

    def _calculate_anls(self, gt, pred, answer_type='string'):
        if len(pred) == 0:
            return 0

        if answer_type == 'not-answerable':
            return 1 if pred in ['', 'none', 'NA', None, []] else 0

        if pred == 'none' and answer_type != 'not-answerable':
            return 0

        answers_similarity = [1 - self.get_edit_distance(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
        max_similarity = max(answers_similarity)

        anls = max_similarity if max_similarity >= self.anls_threshold else 0
        return anls

    def save_state(self, _dir):
        with open(os.path.join(_dir, "evaluator.json"), 'w') as f:
            json.dump({"best_metric": self.best_metric, "best_epoch": self.best_epoch, "curr_epoch": self.curr_epoch, "ascending": self.ascending}, f)

    def load_state(self, _dir):
        with open(os.path.join(_dir, "evaluator.json"), 'r') as f:
            _dict = json.load(f)
            assert set(['best_metric', 'best_epoch', 'curr_epoch', 'ascending']) == _dict.keys()
            self.ascending = _dict["ascending"]
            self.curr_epoch = _dict["curr_epoch"]
            self.best_metric = _dict["best_metric"]
            self.best_epoch = _dict["best_epoch"]

def evaluate_vqa(model, val_dl, tokenizer, args, evaluator, logger):
    per_dset_scores = {_dset: {"per_sample": {}, "loss": [], "score": []} for _dset in args.dataset.split(',')}

    model.cuda()
    model.eval()

    for _bind, _batch in enumerate(val_dl):
        _bs = len(_batch["question_id"])
        src_ids, src_mask = _batch["input_text_id"].cuda(), _batch["input_text_mask"].cuda()
        seg_data = _batch["input_text_box"].cuda()
        image = _batch["image"].cuda()
        visual_seg_data = _batch["image_box"].cuda()
        with torch.no_grad():
            loss = model(
                input_ids=_batch["input_text_id"].cuda(),
                attention_mask=_batch["input_text_mask"].cuda(),
                seg_data=_batch["input_text_box"].cuda(),
                image=_batch["image"].cuda(),
                visual_seg_data=_batch["image_box"].cuda(),
                labels=_batch["label"].cuda(),
            ).loss
            generated_ids = model.generate(
                src_ids,
                attention_mask=src_mask,
                seg_data=seg_data,
                visual_seg_data=visual_seg_data,
                image=image,
                use_cache=True,
                num_beams=args.eval_num_beam,
                max_length=args.eval_max_len,
            )
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        scores = evaluator.get_scores(_batch['answers'], preds, _batch['dataset'])
        if _bind % 20 == 0 or _bind == len(val_dl)-1:
            task = _batch['dataset'][0]
            _scores = []
            for _s in scores:
                if _s["task"] == task:
                    _scores += list(tuple(zip(_s["metric"], _s["score"])))
            _sums = {_k: [] for _k,_ in _scores}
            for _k, _v in _scores:
                _sums[_k] += [_v]

            logger.info(f"[VALIDATION] Epoch[{_bind}] batch data: {task}, loss: {loss.item():.6f}, " +\
                ', '.join([f"{_k}: {np.mean(_sums[_k]):.6f}" for _k,_v in _sums.items()]).strip()
            )

        for _ind in range(_bs):
            _dset, _ques_id = _batch['dataset'][_ind], _batch['question_id'][_ind]
            per_dset_scores[_dset]['per_sample'][_ques_id] = {
                "score": scores[_ind]["score"],
                "pred_answer": preds[_ind],
                "gt_answer": _batch['answers'][_ind],
            }
            per_dset_scores[_dset]["score"].append(scores[_ind]["score"][0])
            per_dset_scores[_dset]["metric"] = scores[_ind]["metric"][0]
        per_dset_scores[_dset]["loss"].append(loss.item())

    model.to('cpu')

    avg_loss = sum([sum(per_dset_scores[_dset]["loss"]) for _dset in args.dataset.split(',')]) / len(val_dl)
    avg_metric = np.mean(
        [ np.mean(_v["score"]) for _v in per_dset_scores.values()]
    )
    return avg_loss, avg_metric, per_dset_scores

def evaluate_ssl(model, val_dl, args, logger):
    model.cuda()
    model.eval()

    xent_loss = nn.CrossEntropyLoss(ignore_index=-100)

    task_loss, epoch_loss = {_task: [] for _task in args.ssl_task.split(',')}, []
    for _bind, _batch in enumerate(val_dl):
        with torch.no_grad():
            agg_loss = []
            for _task in task_loss:
                target_ids =_batch[f"label_{_task}"].cuda()
                decoder_input_ids = model._shift_right(target_ids)
                input_dict = {
                    "input_ids": _batch[f"input_text_id_{_task}"].cuda(),
                    "attention_mask": _batch[f"input_text_mask_{_task}"].cuda(),
                    "seg_data": _batch[f"input_text_box_{_task}"].cuda(),
                    "image": _batch["image"].cuda(),
                    "visual_seg_data": _batch["image_box"].cuda(),
                    "decoder_input_ids": decoder_input_ids,
                }
                lm_logits = model(**input_dict)[0]
                assert lm_logits.size(-1) == model.config.vocab_size
                loss = xent_loss(lm_logits.view(-1, lm_logits.shape[-1]), target_ids.view(-1))
                agg_loss += [loss.mean()]
                task_loss[_task] += [loss.mean().item()]
            agg_loss = torch.stack(agg_loss).mean()

        epoch_loss += [agg_loss.item()] # batch average
        if _bind % 20 == 0 or _bind == len(val_dl)-1:
            logger.info(f"[VALIDATION] Epoch[{_bind}] batch loss: {agg_loss.item():.6f}, " +\
                " ".join([f"{_t}: {_l[-1]:.6f}," for _t,_l in task_loss.items()])
            )

    model.to('cpu')

    valloss = sum(epoch_loss)/len(val_dl) # dataloader average
    return valloss

def parse_args():
    parser = argparse.ArgumentParser(description='DocVQA Evaluation')

    parser.add_argument('--docvqa', default=True, action='store_true')
    parser.add_argument('--ssl_task', default=None)

    parser.add_argument('--dataset', type=str, default='wtq,docvqa,tabfact', help='ordered by priority, delimited by comma.')
    parser.add_argument('--data_dir', type=str, required=False, default="/data/users/vkhanh/due", help="data directory.")

    parser.add_argument('--model_type', type=str, default='UdopUnimodel', help='model')
    parser.add_argument('--model_name_or_path', type=str, default='t5-base', help='(pretrained) model name/path.')
    parser.add_argument('--config_name', type=str, default=None, help='config name/path if different from model_name.')
    parser.add_argument('--tokenizer_name', type=str, default=None, help='tokenizer name/path if different from model_name.')

    parser.add_argument('--mae_version', type=str, default='mae_vit_base_patch16', help='MAE config.')
    parser.add_argument('--mae_checkpoint', type=str, default='mae_ckpt/mae_pretrain_vit_base.pth', help='MAE pre-trained weights.')
    parser.add_argument('--image_size', type=int, default=224, help='size of input image.')
    parser.add_argument('--max_source_len', type=int, default=1024, help='maximum text input length.')
    parser.add_argument('--max_target_len', type=int, default=256, help='maximum text output length.')
    parser.add_argument('--eval_batch_size', type=int, default=32, help="evaluate batch size.")
    parser.add_argument('--eval_num_beam', type=int, default=1, help='num beams on decoding.')
    parser.add_argument('--eval_max_len', type=int, default=256, help='max sequence length on decoding.')

    parser.add_argument('--num_worker', type=float, default=8, help='dataloader worker.')
    parser.add_argument('--init_seed', type=int, default=0, help="random seed.")
    parser.add_argument('--log_dir', type=str, default="save/logs/", help='log directory.')
    parser.add_argument('--log_file', type=str, default=None, help='log name.')
    parser.add_argument('--result_dir', type=str, default="save/result/", help='result directory.')

    args = parser.parse_args()
    return args

def verify_args(args):
    pass

def main():
    args = parse_args()
    verify_args(args)

    dt = datetime.datetime.now()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(args.log_dir, "eval_" + (f"exp_logs_{dt.strftime('%d-%m-%Y-%H-%M-%S')}" if args.log_file is None else args.log_file) + ".log"),
        format='[%(asctime)s] %(levelname)-8s %(message)s',
        datefmt='%d-%m-%Y %I:%M:%S %p', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    with open(os.path.join(args.log_dir, "eval_" + (f"exp_args_{dt.strftime('%d-%m-%Y-%H-%M-%S')}" if args.log_file is None else args.log_file) + ".json"), 'w') as f:
        json.dump(str(args), f)
    logger.info(args)
    logger.info("#" * 100)

    seed = args.init_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    logger.info("Initialize model...")
    global_model, tokenizer = init_models(args, 1)

    evaluator = Evaluator(case_sensitive=False)
    logger.info(f"Task=docvqa")

    test = json.load(open(os.path.join(args.data_dir, f"fldocvqa/test/{'_'.join(args.dataset.split(','))}_allin.json"), 'r'))
    test_globinds = {int(_k):_v for _k, _v in test["partitions"]["docvqa"].items()}
    test_dl_global, test_ds = get_dataloader("test", test_globinds[0], args.eval_batch_size, tokenizer, args)
    logger.info(f"Num TEST global: {len(test_ds)}, TEST steps: {len(test_dl_global)}")

    _, metric, per_dset_scores = evaluate_vqa(global_model, test_dl_global, tokenizer, args, evaluator, logger)
    breakdown = ', '.join([f"{_dset}({np.mean(_v['score']):.4f} {_v['metric']})" for _dset,_v in per_dset_scores.items()])
    logger.info(
        f"[TEST] DocVQA global model: metric {metric:.4f}, " +\
        f"breakdown: {breakdown}"
    )
    per_dset_scores = save_result(per_dset_scores, metric, args)
    logger.info(
        f"Metric: {per_dset_scores['metric']*100:.4f}\t" +\
        "\t".join([f"{_k}({_v['metric']}): {_v['score']*100:.4f}" for _k,_v in per_dset_scores.items() if _k != 'metric'])
    )

    elapsed = datetime.datetime.now() - dt
    logger.info(f"Evaluate time: {elapsed.days} days: {elapsed.seconds//3600} hours: {elapsed.seconds//60%60} mins !!!")

if __name__ == '__main__':
    main()

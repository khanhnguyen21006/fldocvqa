import os, json, copy, random, datetime
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from model import *
from eval import *

def parse_args():
    parser = argparse.ArgumentParser()

    # Task CONFIG
    parser.add_argument('--docvqa', default=False, action='store_true', help='docvqa supervised training or self pretraining.')
    parser.add_argument('--ssl_task', type=str, default='lm,tm,tlm', help='self pretraining tasks (lm,tm,tlm).')
    parser.add_argument('--lm_prob', type=float, default=0.75, help='masked layout modeling (easy).')
    parser.add_argument('--tm_prob', type=float, default=0.5, help='masked text modeling (medium).')
    parser.add_argument('--tlm_prob', type=float, default=0.15, help='joint text-layout modeling (hard).')

    # Data CONFIG
    parser.add_argument('--dataset', type=str, default='wtq,docvqa,tabfact', help='ordered by priority (wtq,docvqa,tabfact), delimited by comma.')
    parser.add_argument('--data_dir', type=str, required=False, default="/data/users/vkhanh/due", help="data directory.")

    # Model CONFIG
    parser.add_argument('--model_type', type=str, default='UdopUnimodel', help='model')
    parser.add_argument('--model_name_or_path', type=str, default='t5-base', help='(pretrained) model name/path.')
    parser.add_argument('--config_name', type=str, default=None, help='config name/path if different from model_name.')
    parser.add_argument('--tokenizer_name', type=str, default=None, help='tokenizer name/path if different from model_name.')
    parser.add_argument('--resume_from', type=str, default=None, help="resume training on start")

    parser.add_argument('--mae_version', type=str, default='mae_vit_base_patch16', help='MAE config.')
    parser.add_argument('--mae_checkpoint', type=str, default='mae_ckpt/mae_pretrain_vit_base.pth', help='MAE pre-trained weights.')
    parser.add_argument('--image_size', type=int, default=224, help='size of input image.')
    parser.add_argument('--max_source_len', type=int, default=1024, help='maximum text input length.')
    parser.add_argument('--max_target_len', type=int, default=256, help='maximum text output length.')

    # FL CONFIG
    parser.add_argument('--algo', type=str, default='fedavg', choices=['fedavg', 'fedprox', 'allin'], help='federated learning algorithm.')
    parser.add_argument('--num_round', type=int, default=10, help='number of maximum communication rounds T.')
    parser.add_argument('--num_client', type=int, default=3, help='number of clients K.')
    parser.add_argument('--num_epoch', type=int, default=1, help='number of local epochs E.')
    parser.add_argument('--sample_prob', type=float, default=0.35, help='fraction of clients per round C.')

    # CLIENTOPT CONFIG
    parser.add_argument('--batch_size', type=int, default=8, help='local training batch size B.')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='local training learning rate nl.')
    parser.add_argument('--optimizer', type=str, default='adamw', help='local optimizer CLIENTOPT.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='local training warmup steps.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='local training weight decay.')
    parser.add_argument('--label_smoothing', type=int, default=0, help='label smoothing.')
    parser.add_argument('--eval_batch_size', type=int, default=32, help="evaluate batch size.")
    parser.add_argument('--eval_num_beam', type=int, default=1, help='num beams on decoding.')
    parser.add_argument('--eval_max_len', type=int, default=256, help='max sequence length on decoding.')
    
    # SERVEROPT CONFIG
    parser.add_argument('--server_optimizer', type=str, default=None, help='server optimizer SERVEROPT.')
    parser.add_argument('--server_learning_rate', type=float, default=1, help='server learning rate ns')
    parser.add_argument('--beta_momentum', type=float, default=0.9, help='server momentum coefficient beta1.')
    parser.add_argument('--beta_rmsprop', type=float, default=0.99, help='server rmsprop coefficient beta2.')
    parser.add_argument('--eps', type=float, default=1e-3, help='server adam epsilon e.')
    parser.add_argument('--bc', default=False, action='store_true', help='bias correction.')

    # MISC
    parser.add_argument('--num_worker', type=float, default=8, help='dataloader worker.')
    parser.add_argument('--init_seed', type=int, default=0, help="random seed.")
    parser.add_argument('--eval_start', default=False, action='store_true', help="evaluation on start.")
    parser.add_argument('--keep_prev_round', default=False, action='store_true', help="keep all round global checkpoints.")
    parser.add_argument('--save_local_per_round', default=False, action='store_true', help="save local per round.")
    parser.add_argument('--save_final_round', default=False, action='store_true', help="save final round.")
    parser.add_argument('--log_dir', type=str, default="save/logs/", help='log directory.')
    parser.add_argument('--log_file', type=str, default=None, help='log name.')
    parser.add_argument('--ckpt_dir', type=str, default="save/models/", help='model checkpoint directory.')

    # parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    # parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    # parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    # parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    # parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    # parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    # parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    # parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')

    args = parser.parse_args()
    return args

def verify_args(args):
    assert args.num_client > 0
    if args.num_client == 1:
        assert args.algo == "allin" and args.sample_prob == 1
    if args.resume_from:
        assert os.path.exists(os.path.join(args.resume_from))
    if not args.docvqa:
        tasks = args.ssl_task.split(',')
        assert len(tasks) > 0
        if 'lm' in tasks: assert args.lm_prob > 0
        if 'tm' in tasks: assert args.tm_prob > 0
        if 'tlm' in tasks: assert args.tlm_prob > 0
    if args.server_optimizer:
        """
            FedAvgM = {'algo': 'fedavg', 'server_optimizer': 'momentum'}
            FedAdam = {'algo': 'fedavg', 'server_optimizer': 'adam'}
        """
        assert args.server_optimizer in ['momentum', 'adam']

def train_local_vqa(model, train_dl, evaluator, tokenizer, logger, args):
    lr, opt, wd = args.learning_rate, args.optimizer, args.weight_decay

    model = nn.DataParallel(model)
    model.cuda()

    if opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    elif opt == 'amsgrad':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd, amsgrad=True)
    if opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=wd)

    for _epoch in range(args.num_epoch):
        epoch_loss, eval_score, eval_count = [], [], 0
        for _bind, _batch in enumerate(train_dl):
            ts = datetime.datetime.now()
            target_ids =_batch["label"].cuda()
            decoder_input_ids = model.module._shift_right(target_ids)
            input_dict = {
                "input_ids": _batch["input_text_id"].cuda(),
                "attention_mask": _batch["input_text_mask"].cuda(),
                "seg_data": _batch["input_text_box"].cuda(),
                "image": _batch["image"].cuda(),
                "visual_seg_data": _batch["image_box"].cuda(),
                "decoder_input_ids": decoder_input_ids,
                # "decoder_attention_mask": label_maskdecoder_attention_mask,
            }
            output = model(**input_dict)

            lm_logits = output[0]
            if args.label_smoothing == 0:
                xe_loss = nn.CrossEntropyLoss(ignore_index=-100)
                assert lm_logits.size(-1) == model.module.config.vocab_size
                loss = xe_loss(lm_logits.view(-1, lm_logits.shape[-1]), target_ids.view(-1))
            else:
                lprobs = F.log_softmax(lm_logits, dim=-1)
                loss, nll_loss = label_smoothed_nll_loss(lprobs, target_ids, args.label_smoothing, ignore_index=-100)
                loss = output.loss
            # print(f"size: {loss.size()}, mean: {loss.mean()}, sum: {loss.sum()}")

            loss.mean().backward()  # batch average
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += [loss.mean().item()]
            bmessage = ""
            if _bind % 20 == 0 or _bind == len(train_dl)-1:
                bmessage = f"[TRAIN] Epoch[{_epoch}][{_bind}] batch loss: {loss.mean().item():.6f}, " +\
                    f"learning rate: {optimizer.param_groups[0]['lr']}, " +\
                    f"time: {(datetime.datetime.now() - ts).total_seconds():.2f}s"
            if _bind % 500 == 0 or _bind == len(train_dl)-1:
                with torch.no_grad():
                    generated_ids = model.module.generate(
                        input_dict["input_ids"],
                        attention_mask=input_dict["attention_mask"],
                        seg_data=input_dict["seg_data"],
                        visual_seg_data=input_dict["visual_seg_data"],
                        image=input_dict["image"],
                        use_cache=True,
                        num_beams=args.eval_num_beam,
                        max_length=args.eval_max_len,
                    )
                preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                scores = evaluator.get_scores(_batch['answers'], preds, _batch["dataset"])

                if args.algo in ["fedavg", "fedprox"]:
                    eval_count += len(_batch)  # per-sample average
                    assert len(set(_batch["dataset"])) == 1
                    _score = [_score['score'][0] for _score in scores]
                    eval_score += _score
                    _score = np.mean(_score)
                    metric_name = scores[0]['metric'][0]
                else:
                    eval_count += 1  # batch average
                    metric_name = "metric"
                    _sums = {_dset:[] for _dset in set(_batch['dataset'])}
                    for _score in scores:
                        _sums[_score['task']] += [_score['score']]
                    _score = np.mean(
                        [ np.mean(_v) for _v in _sums.values()]
                    )
                    eval_score += _score
                bmessage = f"[TRAIN] Epoch[{_epoch}][{_bind}] batch loss: {loss.mean().item():.6f}, " +\
                    f"{metric_name} {_score:.4f}, " +\
                    f"learning rate: {optimizer.param_groups[0]['lr']}, " +\
                    f"time: {(datetime.datetime.now() - ts).total_seconds():.2f}s"

            if bmessage != "":
                logger.info(bmessage)

        trainscore, trainloss = sum(eval_score)/eval_count, sum(epoch_loss)/len(train_dl)  # dataloader average
        logger.info('>'*5 + f' Epoch: {_epoch}, train loss: {trainloss:.4f}, ' +\
            f"train {metric_name}: {trainscore:.4f}, "
            f"learning rate: {optimizer.param_groups[0]['lr']}"
        )

    model.to('cpu')

    return trainscore, trainloss, scores[0]['metric'][0]

def train_local_ssl(model, train_dl, logger, args):
    lr, opt, wd = args.learning_rate, args.optimizer, args.weight_decay  # pre-training config might be different

    model = nn.DataParallel(model)
    model.cuda()

    xent_loss = nn.CrossEntropyLoss(ignore_index=-100)

    if opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    elif opt == 'amsgrad':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd, amsgrad=True)
    if opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=wd)

    for _epoch in range(args.num_epoch):
        task_loss, epoch_loss = {_task: [] for _task in args.ssl_task.split(',')}, []
        for _bind, _batch in enumerate(train_dl):
            ts = datetime.datetime.now()
            agg_loss = []
            for _task in task_loss:
                target_ids =_batch[f"label_{_task}"].cuda()
                decoder_input_ids = model.module._shift_right(target_ids)
                input_dict = {
                    "input_ids": _batch[f"input_text_id_{_task}"].cuda(),
                    "attention_mask": _batch[f"input_text_mask_{_task}"].cuda(),
                    "seg_data": _batch[f"input_text_box_{_task}"].cuda(),
                    "image": _batch["image"].cuda(),
                    "visual_seg_data": _batch["image_box"].cuda(),
                    "decoder_input_ids": decoder_input_ids,
                }
                lm_logits = model(**input_dict)[0]
                assert lm_logits.size(-1) == model.module.config.vocab_size
                loss = xent_loss(lm_logits.view(-1, lm_logits.shape[-1]), target_ids.view(-1))
                agg_loss += [loss.mean()]
                task_loss[_task] += [loss.mean().item()]
            agg_loss = torch.stack(agg_loss).mean()

            agg_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += [agg_loss.item()] # batch average
            if _bind % 20 == 0 or _bind == len(train_dl)-1:
                logger.info(f"[TRAIN] Epoch[{_epoch}][{_bind}] batch loss: {agg_loss.item():.6f}, " +\
                    " ".join([f"{_t}: {_l[-1]:.6f}," for _t,_l in task_loss.items()]) +\
                    f"learning rate: {optimizer.param_groups[0]['lr']}, " +\
                    f"time: {(datetime.datetime.now() - ts).total_seconds():.2f}s"
                )

        trainloss = sum(epoch_loss)/len(train_dl) # dataloader average
        logger.info('>'*5 + f' Epoch: {_epoch}, train loss: {trainloss:.4f}, ' +\
            " ".join([f"{_t}: {np.mean(_l):.6f}," for _t,_l in task_loss.items()]) + ", " +\
            f"learning rate: {optimizer.param_groups[0]['lr']}"
        )

    model.to('cpu')

    return trainloss

def train_local(models, train_inds, val_inds, tokenizer, logger, args, evaluator=None, fl_round=None):
    ### global_model=None, prev_model_pool=None, server_c=None, clients_c=None: params for fedprox|moon ###
    pc_loss, pc_score = [], []

    for _ind, _model in models.items():
        train_dl_local, _ = get_dataloader("train", train_inds[_ind], args.batch_size, tokenizer, args)
        logger.info('*'*10 + ("" if fl_round is None else f"FL round : {fl_round}") + f" training client {_ind} " + '*'*10)
        logger.info(f"Data: {[_k for _k,_v in train_inds[_ind].items() if len(_v) > 0]}" +\
            f", Num TRAIN data: {len(train_dl_local.dataset)}, TRAIN steps: {len(train_dl_local)}"
        )
        if args.algo in ['fedavg', 'local_training']:
            if args.docvqa:
                trainloss, trainscore, trainmetric = train_local_vqa(_model, train_dl_local, evaluator, tokenizer, logger, args)
                pc_loss.append(trainloss)
                pc_score.append(trainscore)
            else:
                trainloss = train_local_ssl(_model, train_dl_local, logger, args)
                pc_loss.append(trainloss)
        logger.info('*'*10 + f' Training client {_ind} complete ' + '*'*10)

    if args.docvqa:
        if args.algo == 'local_training':
            logger.info(
                f"Mean/std over all clients: " +\
                f"loss {np.mean(pc_loss):.4f}{np.std(pc_loss):.4f}, " +\
                f"{trainmetric} {np.mean(pc_score):.4f}{np.std(pc_score):.4f}, "
            )

    return models

def main():
    args = parse_args()
    verify_args(args)

    dt = datetime.datetime.now()

    if args.log_file is not None:
        ver = 0
        while os.path.exists(os.path.join(args.log_dir, args.log_file+".log")):
            args.log_file = args.log_file + f"_v{str(ver+1)}"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(args.log_dir, (f"exp_logs_{dt.strftime('%d-%m-%Y-%H-%M-%S')}" if args.log_file is None else args.log_file) + ".log"),
        format='[%(asctime)s] %(levelname)-8s %(message)s',
        datefmt='%d-%m-%Y %I:%M:%S %p', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    with open(os.path.join(args.log_dir, (f"exp_args_{dt.strftime('%d-%m-%Y-%H-%M-%S')}" if args.log_file is None else args.log_file) + ".json"), 'w') as f:
        json.dump(str(args), f)
    logger.info(args)
    logger.info("#" * 100)

    seed = args.init_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    logger.info("Initialize global/local models...")
    local_models = init_models(args, args.num_client)
    global_model, tokenizer = init_models(args, 1)
    num_param, num_trainable = count_params(global_model)
    logger.info(f"Model parameters: {num_param}" +\
        f", Trainable: {num_trainable} ({num_trainable*100/num_param:.2f}%) "
    )

    if args.resume_from:
        evaluator = Evaluator(case_sensitive=False, ascending=args.docvqa, **vars(args))
        round_start = evaluator.curr_epoch
    else:
        evaluator = Evaluator(case_sensitive=False, ascending=args.docvqa)
        round_start = 0
    logger.info(f"Task={'docvqa' if args.docvqa else f'ssl({args.ssl_task})'}")

    train_globinds, val_globinds, _ = get_partition(args.data_dir, args.dataset, 1, args.docvqa)
    train_dl_global, train_ds = get_dataloader("train", train_globinds[0], args.batch_size, tokenizer, args)
    val_dl_global, val_ds = get_dataloader("val", val_globinds[0], args.eval_batch_size, tokenizer, args)
    logger.info(f"Num TRAIN/VAL global: {len(train_ds)}/{len(val_ds)}, VAL steps: {len(val_dl_global)}")
    if args.num_client > 1:
        train_client2inds, _, dset_stats = get_partition(args.data_dir, args.dataset, args.num_client, args.docvqa)
        logger.info("\t".join([
            f"Dataset {_dset}: {dset_stats[_dset]['num_shard']} shard({dset_stats[_dset]['num_data_per_shard']})" 
            for _dset in dset_stats
        ]))
    num_client_per_round = int(args.num_client * args.sample_prob)
    logger.info(f"Num training client per round: {num_client_per_round}")
    all_clients = [_cli for _cli in range(args.num_client)]
    round_clients = []
    if num_client_per_round != args.num_client:
        round_clients = [random.sample(all_clients, num_client_per_round) for _ in range(round_start, args.num_round)]
    else:
        round_clients = [all_clients for _ in range(round_start, args.num_round)]

    if args.server_optimizer == 'momentum':
        lrg, beta1, bc = args.server_learning_rate, args.beta_momentum, args.bc
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    elif args.server_optimizer == 'adam':
        lrg, beta1, beta2, eps, bc = args.server_learning_rate, args.beta_momentum, args.beta_rmsprop, args.eps, args.bc
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
        rmsprob_v = copy.deepcopy(global_model.state_dict())
        for key in rmsprob_v:
            rmsprob_v[key] = eps**2

    if args.eval_start:
        loss, metric, per_dset_scores = evaluate_vqa(global_model, val_dl_global, tokenizer, args, evaluator, logger)
        breakdown = ', '.join([f"{_dset}({np.mean(_v['score']):.4f} {_v['metric']})" for _dset,_v in per_dset_scores.items()])
        logger.info(
            f"[VALIDATION] docvqa sanity check: metric {metric:.4f}, " +\
            f"breakdown: {breakdown}"
        )

    if args.algo == 'fedavg':
        logger.info(f"Federated TRAINing Algorithm: FedAvg")
        for _round in range(round_start, args.num_round):
            logger.info('='*20 + f" FL round [{str(_round)}]! " + '='*20)
            curr_round_clients = round_clients[_round]
            curr_round_data = [_k for _cli in curr_round_clients for _k,_v in train_client2inds[_cli].items() if len(_v) > 0]
            logger.info('*'*10 + f' this round client(s): {curr_round_clients}, data: {curr_round_data} ' + '*'*10)

            global_w = global_model.state_dict()
            if args.server_optimizer:
                old_w = copy.deepcopy(global_model.state_dict())

            curr_round_models = {_cli: local_models[_cli] for _cli in curr_round_clients}
            for _model in curr_round_models.values():
                _model.load_state_dict(global_w)

            train_local(curr_round_models, train_client2inds, val_globinds, tokenizer, logger, args, evaluator=evaluator, fl_round=_round)

            curr_round_train_total = sum([len(train_client2inds[_cli]) for _cli in curr_round_clients])
            agg_w = [len(train_client2inds[_cli])/curr_round_train_total for _cli in curr_round_clients]
            for _w_ind, _model in enumerate(curr_round_models.values()):
                _curr_param = _model.state_dict()
                if _w_ind == 0:
                    for _p in _curr_param:
                        global_w[_p] = _curr_param[_p] * agg_w[_w_ind]
                else:
                    for _p in _curr_param:
                        global_w[_p] += _curr_param[_p] * agg_w[_w_ind]
            if args.server_optimizer == 'momentum':
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = beta1 * moment_v[key] + (1 - beta1) * delta_w[key]
                    if bc:
                        bias_coeff = 1-beta1**(_round+1)
                        logger.info(f"[TRAIN] Epoch[{_round}] FedAvgM, bias: {bias_coeff}")
                        moment_v[key] = torch.div(moment_v[key], bias_coeff)
                    global_w[key] = old_w[key] - lrg * moment_v[key]
            elif args.server_optimizer == 'adam':
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = beta1 * moment_v[key] + (1 - beta1) * delta_w[key]
                    rmsprob_v[key] = beta2 * rmsprob_v[key] + (1 - beta2) * delta_w[key]**2
                    if bc:
                        bias_coeff1 = 1-beta1**(_round+1)
                        bias_coeff2 = 1-beta2**(_round+1)
                        logger.info(f"[TRAIN] Epoch[{_round}] FedAdam, bias 1: {bias_coeff1}, bias 2: {bias_coeff2}")
                        moment_v[key] = torch.div(moment_v[key], bias_coeff1)
                        rmsprob_v[key] = torch.div(rmsprob_v[key], bias_coeff2)
                    global_w[key] = old_w[key] - torch.div(lrg * moment_v[key], torch.add(rmsprob_v[key]**0.5, eps))

            global_model.load_state_dict(global_w)

            if args.docvqa:
                loss, metric, per_dset_scores = evaluate_vqa(global_model, val_dl_global, tokenizer, args, evaluator, logger)
                is_updated = evaluator.update_global_metrics(metric, _round)
                breakdown = ', '.join([f"{_dset}({np.mean(_v['score']):.4f} {_v['metric']})" for _dset,_v in per_dset_scores.items()])
                logger.info(
                    f"[VALIDATION] round[{str(_round)}], data{curr_round_data} global model: loss {loss:.4f},\tmetric {metric:.4f},\t" +\
                    f"breakdown: {breakdown}" +\
                    ("\tBest Performance!" if is_updated else "")
                )
            else:
                valloss = evaluate_ssl(global_model, val_dl_global, args, logger)
                is_updated = evaluator.update_global_metrics(valloss, _round)
                logger.info(
                    f"[VALIDATION] round[{str(_round)}] data{curr_round_data} global model: val loss {valloss:.4f}" +\
                    ("\tBest Performance!" if is_updated else "")
                )
            save_model(global_model, tokenizer, evaluator, "global", args, _round, update_best=is_updated, keep_prev_round=args.keep_prev_round)
            if args.save_local_per_round:
                for _cli, _model in local_models.items():
                    save_model(_model, tokenizer, evaluator, f"local{_cli}", args, _round, keep_prev_round=args.keep_prev_round)
            else:
                if _round == args.num_round-1 and args.save_final_round:
                    for _cli, _model in local_models.items():
                        save_model(_model, tokenizer, evaluator, f"final_local{_cli}", args, _round)

    elif args.algo == 'local_training':
        logger.info("TRAINing Algorithm: local training")
        train_local(curr_round_models, train_client2inds, val_globinds, tokenizer, args, evaluator=evaluator)
        if args.save_local_per_round:
            for _cli, _model in local_models.items():
                save_model(_model, tokenizer, evaluator, f"local{_cli}", args, _round, keep_prev_round=args.keep_prev_round)
        else:
            if _round == args.num_round-1 and args.save_final_round:
                for _cli, _model in local_models.items():
                    save_model(_model, tokenizer, evaluator, f"final_local{_cli}", args, _round)

    elif args.algo == 'allin':
        logger.info("TRAINing Algorithm: allin")
        for _epoch in range(round_start, args.num_round):
            if args.docvqa:
                train_local_vqa(global_model, train_dl_global, evaluator, tokenizer, logger, args)
                loss, metric, per_dset_scores = evaluate_vqa(global_model, val_dl_global, tokenizer, args, evaluator, logger)
                is_updated = evaluator.update_global_metrics(metric, _epoch)
                breakdown = ', '.join([f"{_dset}({np.mean(_v['score']):.4f} {_v['metric']})" for _dset,_v in per_dset_scores.items()])
                logger.info(
                    f"[VALIDATION] Epoch[{str(_epoch)}] global model: loss {loss:.4f},\tmetric {metric:.4f},\t" +\
                    f"breakdown: {breakdown}" +\
                    ("\tBest Performance!" if is_updated else "")
                )
            else:
                trainloss = train_local_ssl(global_model, train_dl_global, logger, args)
                valloss = evaluate_ssl(global_model, val_dl_global, args, logger)
                is_updated = evaluator.update_global_metrics(valloss, _epoch)
                logger.info(
                    f"[VALIDATION] Epoch[{str(_epoch)}] global model: train loss {trainloss:.4f},\tval loss {valloss:.4f}" +\
                    ("\tBest Performance!" if is_updated else "")
                )
            save_model(global_model, tokenizer, evaluator, "global", args, _epoch, update_best=is_updated, keep_prev_round=args.keep_prev_round)

    elapsed = datetime.datetime.now() - dt
    logger.info(f"Training finished after: {elapsed.days} days: {elapsed.seconds//3600} hours: {elapsed.seconds//60%60} mins !!!")

if __name__ == '__main__':
    main()

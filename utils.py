import os, json, re, random, shutil
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
# sns.set_style('darkgrid')

import torch
from torchvision import transforms as t

########## MISC utils ##########
def get_partition(data_root, dataset, num_client, docvqa):
	data_dir = os.path.join(data_root, "fldocvqa")
	DATASETS = [_dset for _dset in dataset.split(',')]
	dset_name = '_'.join(DATASETS)
	task = "docvqa" if docvqa else "ssl"

	par_name = f"{dset_name}_nonidd{num_client}" if num_client > 1 else f"{dset_name}_allin"
	if os.path.isfile(os.path.join(data_dir, f"train/{par_name}.json")):
		train = json.load(open(os.path.join(data_dir, f"train/{par_name}.json"), 'r'))
		val = json.load(open(os.path.join(data_dir, f"val/{dset_name}_allin.json"), 'r'))
		return {int(_k):_v for _k, _v in train["partitions"][task].items()}, {int(_k):_v for _k, _v in val["partitions"][task].items()}, train["dset_stat"]
	else:
		if num_client == 1:
			train_inds, val_inds, num_shard_per_dset = {}, {}, {}
			for _split in ["val", "test", "train"]:
				partitions = {"docvqa": {0: {_dset: [] for _dset in DATASETS}}, "ssl": {0: {_dset: [] for _dset in DATASETS}}}
				for _dset in DATASETS:
					data_vqa = np.load(os.path.join(data_root, _dset, _split, 'vqa.npy'), allow_pickle=True)
					data_ssl = json.load(open(os.path.join(data_root, _dset, _split, 'ssl.json'), 'r'))
					partitions["docvqa"][0][_dset] = np.arange(len(data_vqa)).tolist()
					partitions["ssl"][0][_dset] = np.arange(len(data_ssl)).tolist()
				with open(os.path.join(data_dir, _split, f'{dset_name}_allin.json'), 'w') as f:
					json.dump({"partitions": partitions, "dset_stat": num_shard_per_dset}, f)
				if _split == "train": train_inds = partitions[task];
				if _split == "val": val_inds = partitions[task];
		else:
			val_inds = json.load(open(os.path.join(data_dir, f"val/{dset_name}_allin.json"), 'r'))["partitions"][task]
			DATASSL = {
				_dset: json.load(open(os.path.join(data_root, _dset, "train/ssl.json"), 'r')) for _dset in DATASETS
			}
			partitions = {
				"docvqa": {_ind: {_dset: [] for _dset in DATASETS} for _ind in range(num_client)},
				"ssl": {_ind: {_dset: [] for _dset in DATASETS} for _ind in range(num_client)},
			}

			total_size = sum([len(_v) for _v in DATASSL.values()])
			shard_size = total_size/num_client
			num_shard_per_dset = {_dset: {"num_shard": int(len(DATASSL[_dset])//shard_size)} for _dset in DATASETS}

			residual = num_client - sum([_v["num_shard"] for _v  in num_shard_per_dset.values()])
			_r = 0
			while _r < residual:
				num_shard_per_dset[DATASETS[_r]]["num_shard"] += 1
				_r += 1
			for _dset in DATASETS:
				num_shard_per_dset[_dset].update({"num_data_per_shard": len(DATASSL[_dset])//num_shard_per_dset[_dset]["num_shard"]})
			assert sum([_v["num_shard"] for _v  in num_shard_per_dset.values()]) == num_client
			inds_per_dset = {}
			for _ind, _dset in enumerate(DATASETS):
				start_ind = 0 if _ind == 0 else sum([num_shard_per_dset[__ds]["num_shard"] for __ds in DATASETS[:_ind]])
				inds_per_dset.update({_dset: non_iid(num_shard_per_dset[_dset]["num_shard"], len(DATASSL[_dset]), start_ind)})
			assert sum([len(inds_per_dset[_dset].keys()) for _dset in DATASETS]) == num_client
			assert sum([sum([len(_v) for _v in inds_per_dset[_dset].values()]) for _dset in DATASETS]) == total_size

			for _ind in range(num_client):
				for _dset in DATASETS:
					if _ind in inds_per_dset[_dset]:
						partitions["ssl"][_ind][_dset] = inds_per_dset[_dset][_ind]
						for _ipd in inds_per_dset[_dset][_ind]:
							partitions["docvqa"][_ind][_dset] += DATASSL[_dset][_ipd]["vqa_indices"]
			with open(os.path.join(data_dir, f'train/{par_name}.json'), 'w') as f:
				json.dump({"partitions": partitions, "dset_stat": num_shard_per_dset}, f)

			train_inds = partitions[task]
		return train_inds, val_inds, num_shard_per_dset

def non_iid(num_shard, data_len, start_ind=0):
	num_data = int(data_len/num_shard)  # num documents per shard
	shard_dict = {_i: [] for _i in range(start_ind, start_ind+num_shard)}  # {0:[], 1:[], ..., n_shard:[]}
	all_data_inds = [_i for _i in range(data_len)]  # [0, 1, 2, ..., data_len]

	# divide and assign {num_data} indices to {num_shard} shards
	for _i in range(start_ind, start_ind+num_shard):
		rand_set = set(random.sample(all_data_inds, num_data))
		shard_dict[_i] = list(rand_set)
		all_data_inds = list(set(all_data_inds) - rand_set)
	shard_dict[_i] += all_data_inds
	return shard_dict

def save_model(model, tokenizer, evaluator, name, args, fl_round, update_best=False, keep_prev_round=False, optimizer=None, scheduler=None):
	save_dir = os.path.join(args.ckpt_dir, args.log_file)
	ckpt_name = f"round{fl_round}_{name}.ckpt"

	model.save_pretrained(os.path.join(save_dir, ckpt_name))
	tokenizer.save_pretrained(os.path.join(save_dir, ckpt_name))
	evaluator.save_state(os.path.join(save_dir, ckpt_name))
	if not keep_prev_round:
		for _prev_round in range(fl_round):
			_prev_round_dir = os.path.join(save_dir, f"round{_prev_round}_{name}.ckpt")
			if os.path.exists(_prev_round_dir):
				shutil.rmtree(_prev_round_dir)

	# torch.save({
	# 	'epoch': fl_round,
	# 	'optimizer': optimizer.state_dict(),
	# 	'lr_scheduler': scheduler,
	# 	}, os.path.join(save_dir, (f"round{fl_round}_" if fl_round else "") + f"{name}.ckpt")
	# )
	if update_best:
		model.save_pretrained(os.path.join(save_dir, "best.ckpt"))
		tokenizer.save_pretrained(os.path.join(save_dir, "best.ckpt"))
		evaluator.save_state(os.path.join(save_dir, "best.ckpt"))

def save_result(per_dset_scores, metric, args):
	save_dir = os.path.join(args.result_dir, args.log_file)
	for _v in per_dset_scores.values():
		_v.pop("loss")
		_score = _v.pop("score")
		_v["score"] = np.mean(_score)
	per_dset_scores["metric"] = metric
	os.makedirs(save_dir, exist_ok=True)
	with open(os.path.join(save_dir, "vqa_result.json"), 'w') as f:
		json.dump(per_dset_scores, f)
	return per_dset_scores

def count_params(model):
	num_param = sum(p.numel() for p in model.parameters())
	num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return num_param, num_trainable

def plot_single_val_metric_breakdown(expt, title, x='round', ext='png'):
	"""
		Plot val metric breakdown for ONE CONFIG (K,C)
	"""
	labels, colors, markers = ["wtq", "docvqa", "tabfact"], ['#000093', '#009300', '#930000'], ['X', '^', 'P']
	ROOT = "save/logs/"

	x_val, vals= [], [[],[],[]]
	with open(os.path.join(ROOT, expt + ".log")) as f:
		f = f.readlines()
	count = 0
	for _line in f:
		if "global model" in _line:
			_y_wtq = re.search(r"wtq\((\d+\.\d+) anls\)", _line).group(1)
			_y_dv = re.search(r"docvqa\((\d+\.\d+) anls\)", _line).group(1)
			_y_tf = re.search(r"tabfact\((\d+\.\d+) accuracy\)", _line).group(1)
			if x == 'round':
				m = re.search(r'round\[(\d+)\]', _line)
				_x = m.group(1) if m else count
				count += 1
			else:
				raise "Invalid x"
			x_val.append(int(_x))
			vals[0].append(float(_y_wtq))
			vals[1].append(float(_y_dv))
			vals[2].append(float(_y_tf))
	assert len(x_val) == len(vals[0]) == len(vals[1]) == len(vals[2])
	sns.lineplot(x=x_val, y=vals[0], color=colors[0], marker=markers[0], markersize=8)
	sns.lineplot(x=x_val, y=vals[1], color=colors[1], marker=markers[1], markersize=8)
	sns.lineplot(x=x_val, y=vals[2], color=colors[2], marker=markers[2], markersize=8)

	# Add labels and title
	plt.xticks(np.arange(0, len(x_val), 5))

	# fig.suptitle(title)
	plt.xlabel('communication round', weight="bold", fontsize=14)
	plt.ylabel('per-dataset metric', weight="bold", fontsize=14)
	# Add a legend
	plt.legend(
		handles=[Line2D([], [], color=_c, label=_l, marker=_m) for _l,_c,_m in zip(labels, colors, markers)],
		loc='upper right',
		# bbox_to_anchor=(1, 1)
	)
	plt.tight_layout()
	plt.savefig(os.path.join(ROOT, "../figures", f"{title.replace(',', ' ').replace(' ', '_')}_breakdown.{ext}"), bbox_inches='tight')

def plot_single_train_curve(expt, clients, title, ext='png'):
	"""
		Plot training loss curve for ONE CONFIG (K,C) (multi subplots: each client has one curve)
		params: title for (K,C)
	"""
	ROOT = "save/logs/"
	K = len(clients); assert K in [3, 10, 30]
	if K == 3: row=3;col=1;
	if K == 10: row=10;col=1;
	if K == 30: row=5;col=6;

	fig, axs = plt.subplots(nrows=row, ncols=col, figsize=(20,12))
	if col==1: axs = [[_ax] for _ax in axs];
	x_val, loss_val = [], [[] for _ in range(K)] # met_val = [[] for _ in range(K)]
	with open(os.path.join(ROOT, expt + ".log")) as f:
		f = f.readlines()

	start_flag = False
	curr_step, curr_round, curr_client = 0, 0, 0
	for _line in f:
		if "Federated TRAINing Algorithm: FedAvg" in _line and not start_flag:
			start_flag = True
			continue
		if start_flag:
			_g1 = re.search(r'\*+FL round : (\d+) training client (\d+) \*+', _line)
			if _g1:
				curr_round = int(_g1.group(1))
				curr_client = int(_g1.group(2))

			# _y_met = re.search(r"metric (\d+\.\d+)", _line).group(1)
			_g2 = re.search(r'\[TRAIN\] Epoch\[\d+\]\[(\d+)\] batch loss: (\d+\.\d+),', _line)
			if _g2: 
				curr_step += 1;
				x_val.append(curr_step)
				for _k in range(K):
					if _k == curr_client:
						loss_val[_k].append(float(_g2.group(2)))
					else:
						loss_val[_k].append(0.0) 
				# met_val.append(float(_y_met))
	
	assert not any([len(x_val) != len(loss_val[_k]) for _k in range(K)]) # == len(met_val)
	for _k in range(K):
		_i, _j = _k//col, _k%col
		sns.lineplot(x=x_val, y=loss_val[_k], color='#000093', ax=axs[_i][_j])  #, marker='^', markersize=15
		# sns.lineplot(x=x_val, y=met_val, color='#000093', marker='^', ax=ax2)  #, markersize=15

		# Add labels and title
		# axs[_i][_j].set_xticks(np.arange(0, len(x_val), 100))
		axs[_i][_j].set_xlabel('step', fontsize=12, weight='bold')
		axs[_i][_j].set_ylabel(f'loss', fontsize=12, weight='bold')
		axs[_i][_j].set_title(clients[_k], fontsize=12, weight='bold')
		# ax2.set_xticks(np.arange(0, len(x_val), 5))  #, 5
		# ax2.set_xlabel('communication round', fontsize=12, weight='bold')
		# ax2.set_ylabel(f'validation metric', fontsize=12, weight='bold')

	fig.suptitle(title)

	fig.tight_layout()
	plt.savefig(os.path.join(ROOT, "../figures", f"{title.replace(',', ' ').replace(' ', '_')}.{ext}"), bbox_inches='tight')

def plot_single_val_curve_both(expt, title, x='round', ext='png'):
	"""
		Plot val curve for ONE CONFIG (K,C) (2 subplots: 1 for loss, 1 for metric)
		params: title for (K,C)
	"""
	ROOT = "save/logs/"

	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,3.5))
	x_val, loss_val, met_val = [], [], []
	with open(os.path.join(ROOT, expt + ".log")) as f:
		f = f.readlines()
	count = 0
	for _line in f:
		if "global model" in _line:
			_y_loss = re.search(r"loss (\d+\.\d+)", _line).group(1)
			_y_met = re.search(r"metric (\d+\.\d+)", _line).group(1)
			if x == 'round':
				m = re.search(r'round\[(\d+)\]', _line)
				_x = m.group(1) if m else count
				count += 1
			else:
				raise "Invalid x"
			x_val.append(int(_x))
			loss_val.append(float(_y_loss))
			met_val.append(float(_y_met))
	assert len(x_val) == len(loss_val) == len(met_val)
	sns.lineplot(x=x_val, y=loss_val, color='#000093', marker='^', ax=ax1)  #, markersize=15
	sns.lineplot(x=x_val, y=met_val, color='#000093', marker='^', ax=ax2)  #, markersize=15
	ax2.axhline(y=0.5316, linewidth=2, color='orange', ls=':')

	# Add labels and title
	ax1.set_xticks(np.arange(0, len(x_val), 5))  #, 5
	ax1.set_xlabel('communication round', fontsize=12, weight='bold')
	ax1.set_ylabel(f'validation loss', fontsize=12, weight='bold')
	ax2.set_xticks(np.arange(0, len(x_val), 5))  #, 5
	ax2.set_xlabel('communication round', fontsize=12, weight='bold')
	ax2.set_ylabel(f'validation metric', fontsize=12, weight='bold')

	fig.suptitle(title)

	fig.tight_layout()
	plt.savefig(os.path.join(ROOT, "../figures", f"{title.replace(',', ' ').replace(' ', '_')}.{ext}"), bbox_inches='tight')

	return x_val, loss_val, met_val

def plot_single_val_curve_one_metric(expt, title, metric='loss', x='round', ext='png'):
	"""
		Plot val curve for ONE CONFIG (K,C) (1 plots: for loss for metric)
		params: title for (K,C)
	"""
	ROOT = "save/logs/"

	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,3.5))
	x_val, y_val
	with open(os.path.join(ROOT, expt + ".log")) as f:
		f = f.readlines()
	count = 0
	for _line in f:
		if "global model" in _line:
			if metric == 'loss':
				_y_val = re.search(r"loss (\d+\.\d+)", _line).group(1)
			else:
				_y_val = re.search(r"metric (\d+\.\d+)", _line).group(1)
			if x == 'round':
				m = re.search(r'round\[(\d+)\]', _line)
				_x = m.group(1) if m else count
				count += 1
			else:
				raise "Invalid x"
			x_val.append(int(_x))
			y_val.append(float(_y_val))
	assert len(x_val) == len(y_val)
	sns.lineplot(x=x_val, y=y_val, color='#000093', marker='^', ax=ax1)  #, markersize=15
	if metric == 'metric':
		plt.axhline(y=0.5316, linewidth=2, color='orange', ls=':')

	plt.xticks(np.arange(0, len(x_val), 5))
	plt.xlabel('communication round', fontsize=12, weight='bold')
	plt.ylabel(f'validation {metric}', fontsize=12, weight='bold')
	plt.title(title)

	fig.tight_layout()
	plt.savefig(os.path.join(ROOT, "../figures", f"{title.replace(',', ' ').replace(' ', '_')}.{ext}"), bbox_inches='tight')

def plot_multi_val_curves_both(expts, labels, colors, markers, title, x='round', ext='png'):
	"""
		Plot multiple val curves (2 subplots: 1 for loss, 1 for metric)
		params: labels, colors, markers for different C
	"""
	ROOT = "save/logs/"
	assert len(expts) == len(labels)

	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,3.5))
	for _ind, (_expt, _label, _color, _marker) in enumerate(zip(expts, labels, colors, markers)):
		x_val, loss_val, met_val = [], [], []
		with open(os.path.join(ROOT, _expt + ".log")) as f:
			f = f.readlines()
		count = 0
		for _line in f:
			if "global model" in _line:
				_y_loss = re.search(r"loss (\d+\.\d+)", _line).group(1)
				_y_met = re.search(r"metric (\d+\.\d+)", _line).group(1)
				if x == 'round':
					m = re.search(r'round\[(\d+)\]', _line)
					_x = m.group(1) if m else count
					count += 1
				else:
					raise "Invalid x"
				x_val.append(int(_x))
				loss_val.append(float(_y_loss))
				met_val.append(float(_y_met))
		assert len(x_val) == len(loss_val) == len(met_val)
		sns.lineplot(x=x_val, y=loss_val, label=_label, color=_color, marker=_marker, ax=ax1, legend=False)  # legend label
		sns.lineplot(x=x_val, y=met_val, label=_label, color=_color, marker=_marker, ax=ax2, legend=False)  # legend label

	# Add labels and title
	ax1.set_xticks(np.arange(0, len(x_val), 5))
	ax1.set_xlabel('communication round', fontsize=12, weight='bold')
	ax1.set_ylabel(f'validation loss', fontsize=12, weight='bold')
	ax2.set_xticks(np.arange(0, len(x_val), 5))
	ax2.set_xlabel('communication round', fontsize=12, weight='bold')
	ax2.set_ylabel(f'validation metric', fontsize=12, weight='bold')
	ax2.axhline(y=0.5316, linewidth=2, color='orange', ls=':')

	# Add a legend
	fig.legend(
		handles=[Line2D([], [], color=_c, label=_l, marker=_m) for _l,_c,_m in zip(labels, colors, markers)],
		loc='upper center',
		ncol=3,
		frameon=False,
		bbox_to_anchor=(0.5,1.05),
	)
	fig.tight_layout()
	plt.savefig(os.path.join(ROOT, "../figures", f"{title.replace(',', ' ')}.{ext}"), bbox_inches='tight')

def plot_multi_val_curves_one_metric(expts, labels, colors, markers, title, metric='loss', x='round', ext='png'):
	"""
		Plotmultiple  val curves (1 plot for metric(loss or docvqa metric))
		params: labels, colors, markers
	"""
	ROOT = "save/logs/"
	assert len(expts) == len(labels)
	
	for _ind, (_expt, _label, _color, _marker) in enumerate(zip(expts, labels, colors, markers)):
		x_val, y_val = [], []
		with open(os.path.join(ROOT, _expt + ".log")) as f:
			f = f.readlines()
		count = 0
		for _line in f:
			if "global model" in _line:
				if metric == 'loss':
					_y_val = re.search(r"loss (\d+\.\d+)", _line).group(1)
				else:
					_y_val = re.search(r"metric (\d+\.\d+)", _line).group(1)
				if x == 'round':
					m = re.search(r'round\[(\d+)\]', _line)
					_x = m.group(1) if m else count
					count += 1
				else:
					raise "Invalid x"
				x_val.append(int(_x))
				y_val.append(float(_y_val))
		assert len(x_val) == len(y_val)
		sns.lineplot(x=x_val, y=y_val, color=colors[_ind], marker=markers[_ind], legend=False)
		
	# Add labels and title
	if metric == 'metric':
		plt.axhline(y=0.5316, linewidth=2, color='orange', ls=':')
	plt.xticks(np.arange(0, len(x_val), 5))
	plt.xlabel('communication round', fontsize=12, weight='bold')
	plt.ylabel(f'validation {metric}', fontsize=12, weight='bold')
	plt.title(title)
	# Add a legend
	plt.legend(
		handles=[Line2D([], [], color=_c, label=_l, marker=_m) for _l,_c,_m in zip(labels, colors, markers)],
		loc='upper right',
		# ncol=3,
		frameon=False,
		# bbox_to_anchor=(0.5,1.05),
	)
	plt.tight_layout()
	plt.savefig(os.path.join(ROOT, "../figures", f"{title.replace(',', ' ')}.{ext}"), bbox_inches='tight')

########## DATA utils ##########
def get_image_bbox(im_size=224, p=16):
	feature_shape = [im_size//p, im_size//p]
	visual_bbox_x = (torch.arange(
		0,
		1.0 * (feature_shape[1] + 1),
		1.0,
	) / feature_shape[1])
	visual_bbox_y = (torch.arange(
		0,
		1.0 * (feature_shape[0] + 1),
		1.0,
	) / feature_shape[0])
	visual_bbox_input = torch.stack(
		[
			visual_bbox_x[:-1].repeat(feature_shape[0], 1),
			visual_bbox_y[:-1].repeat(feature_shape[1], 1).transpose(0, 1),
			visual_bbox_x[1:].repeat(feature_shape[0], 1),
			visual_bbox_y[1:].repeat(feature_shape[1], 1).transpose(0, 1),
		],
		dim=-1,
	).view(-1, 4)
	return visual_bbox_input

def get_image_transform(im_size=224):
	return t.Compose([
		t.Resize([im_size, im_size]),
		t.ToTensor(),
		t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

def normalize_bbox(bbox, size, scale=1000):
	return [
		int(clamp((scale * bbox[0] / size[0]), 0, scale)),
		int(clamp((scale * bbox[1] / size[1]), 0, scale)),
		int(clamp((scale * bbox[2] / size[0]), 0, scale)),
		int(clamp((scale * bbox[3] / size[1]), 0, scale))
	]

def clamp(num, min_value, max_value):
	return max(min(num, max_value), min_value)

def mask_image_mae():
	for k in ['image_mask_label']:
		if k in features[0] and features[0][k] is not None:
			image_size = batch['image'].size()
			mask_ratio = (random.random() * 0.25 + 0.75) * 0.999
			image_mask_labels = []
			ids_restores = []
			ids_keeps = []
			for d in features:
				mask, ids_restore, ids_remove, ids_keep = random_masking(int(image_size[2]**2/16**2), mask_ratio)
				image_mask_labels.append(mask)
				ids_restores.append(ids_restore)
				ids_keeps.append(ids_keep)
			stack_labels = torch.stack(image_mask_labels, dim=0)
			batch.update({'image_mask_label': stack_labels})
			stack_labels = torch.stack(ids_restores, dim=0)
			batch.update({'ids_restore': stack_labels})
			stack_labels = torch.stack(ids_keeps, dim=0)
			batch.update({'ids_keep': stack_labels})

########## MODEL utils ##########
def pad_sequence(seq, target_len, pad_value=0):
	if isinstance(seq, torch.Tensor):
		n = seq.shape[0]
	else:
		n = len(seq)
		seq = torch.tensor(seq)
	m = target_len - n
	if m > 0:
		ret = torch.stack([pad_value] * m).to(seq)
		seq = torch.cat([seq, ret], dim=0)
	return seq[:target_len]

########## TRAINING utils ##########
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
	"""From fairseq"""
	if target.dim() == lprobs.dim() - 1:
		target = target.unsqueeze(-1)
	nll_loss = -lprobs.gather(dim=-1, index=target)
	smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
	if ignore_index is not None:
		pad_mask = target.eq(ignore_index)
		nll_loss.masked_fill_(pad_mask, 0.0)
		smooth_loss.masked_fill_(pad_mask, 0.0)
	else:
		nll_loss = nll_loss.squeeze(-1)
		smooth_loss = smooth_loss.squeeze(-1)

	nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
	smooth_loss = smooth_loss.sum()
	eps_i = epsilon / lprobs.size(-1)
	loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
	return loss, nll_loss

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# parser.add_argument('--dataset', type=str, default="docvqa", help='dataset')
	# parser.add_argument('--num_client', type=int, default=10, help='dataset')
	# args = parser.parse_args()
	# dset = args.dataset
	# n_client = args.num_client
	# assert dset in DATASETS and n_client > 0

	# expts = ['fedadam09ns0001nl000005eps1e8_noniid10_prob035_r10', 'fedadam09ns0001nl000005eps1e8bc_noniid10_prob035_r10'] # 
	# labels = ['-', 'bias correction'] # 
	# colors = ['green', 'blue'] # 
	# markers = ['X', '^'] # 
	# plot_val_curves(expts, labels, colors, markers, 'val_curve')  #, 'K=3 C=[0.35,0.70,1.00]'

	# expt = 'fedavg_noniid3_prob035'
	# clients = ['wtq', 'docvqa', 'tabfact']  # clients = ['wtq'] + ['docvqa']*4 + ['tabfact']*5
	# plot_single_train_curve(expt, clients, 'FedAvg (K3,C0.35,T10)')

	# expt = 'fedavg_noniid3_prob035_r35'
	# plot_single_val_metric_breakdown(expt, 'FedAvg (K3,C0.35,T35)')

	# expt = 'fedavg_noniid3_prob035_r35'
	# plot_single_val_curve_both(expt, 'FedAvg (K3,C0.35,T35)')

	expts = [
		'fedavg_noniid10_sslall_prob035_r35', 
		'fedadam09ns0001nl000005eps1e8_noniid10_sslall_prob035_r35',
	] # 
	labels = ['fedavg', 'fedadam']  # 
	colors = ['green', 'blue']  # , 'red', 'purple'
	markers = ['X', '^']  # , 'P', 'o'
	plot_multi_val_curves_one_metric(expts, labels, colors, markers, 'val_curves_ssl_K10_C035_T35')


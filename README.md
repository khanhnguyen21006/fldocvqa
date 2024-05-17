# Federated Document Visual Question Answering

http://arxiv.org/abs/2405.06636

## Set-up environment
This code works in our local environment with `CUDA 12.1` and `NVIDIA A40 GPUs`.

```bash
conda env create -f environment.yml
conda activate fldocvqa
```

## Dataset
We use a data split provided from [Duebenchmark](https://github.com/due-benchmark/baselines). Please follow its procedure to download the data (including the PDFs) and maintain the folder structure for each dataset (named in lowercase) as follows:
```bash
└── DATA_ROOT
	└── docvqa
	    ├── val
	    │   ├── document.jsonl
	    │   └── documents_content.jsonl
	    ├── document.jsonl
	    ├── documents.json
	    ├── documents.jsonl
	    ├── documents_content.jsonl
	    ├── test
	    │   ├── document.jsonl
	    │   └── documents_content.jsonl
	    ├── train
	    │   ├── document.jsonl
	    │   └── documents_content.jsonl
	    ├── pdfs
```
To convert DUE data to our format and preprocess, run:
```bash
python convert_due.py --data_dir /data/to/DATA_ROOT --dataset dataset_name
```

## Federated Training
The following are the main FL hyperparameters typically considered in all FL algorithms implemented within our framework, using the same notation as in [FedAvg](https://arxiv.org/abs/1602.05629)

| Argument | Values |
|------|------|
| `--algo` | FL algorithm (). |
| `--num_round` | number of maximum communication rounds T. |
| `--num_client` | number of participating clients K. |
| `--num_epoch` | number of local epochs per round E. |
| `--sample_prob` | random fraction of clients per round C. |

SERVEROPT

CLIENTOPT

Note the code now is restricted by, we will release another version for a more modular regarding optimization algortim
Please run the `python train.py -h` for more options and details.

-----------------------------------------------------
For example, consider a FL setting with `K=10`, `C=0.7`, `T=10`

To pretrain with FPS, run:
```bash
python train.py \
	--ssl_task lm,tm,tlm \
	--algo fedavg \
	--server_optimizer adam \
	--server_learning_rate 0.001 \
	--learning_rate 0.00005 \
	--eps 1e-8 \
	--beta_momentum 0.9 \
	--beta_rmsprop 0.99 \
	--num_client 10 \
	--sample_prob 0.7 \
	--num_round 10 \
	--log_file log/file/name
```
To finetune with FedAvg, run:
```bash
python train.py \
	--docvqa \
	--algo fedavg \
	--num_client 10 \
	--sample_prob 0.7 \
	--num_round 10 \
	--log_file log/file/name
```

## Evaluation
The FL evaluation follows the global scheme using shared holdout set located at the server. The metric is compute as
a two-step average: average the test score on each dataset, and then average per-dataset scores to obtain the final score 
```bash
python eval.py --model_name_or_path /path/to/checkpoint/ --eval_batch_size 32 --log_file log/file/name
```

## Hyperparameter Tuning
Hyperparameter optimization in Federated Learning is crucial yet challenging, as it involves numerous hyperparameters and is costly to tune for tasks like DocVQA. The results reported in the paper are not from exhaustive tuning. We welcome feedback if better results are obtained through more thorough tuning strategies.

## Acknowledgments
This codebase structure is based from [MOON](https://github.com/QinbinLi/MOON) repos.

-----------------------------------------------------
For details of the experiments and results, please refer to our paper. 

```bibtex

```
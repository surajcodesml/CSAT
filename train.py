import os
import re
import gc
import pdb
import torch
import pickle
import random
import argparse
import datetime
import numpy as np
import yaml
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from torchvision.ops import complete_box_iou_loss
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from pretrain import *
from utils.plots import plot_images
from utils.dataloader import create_dataloader
from loss.matching_loss import build_matcher
from utils.general import xywh2xyxy, xyxy2xywh
from utils.util import *
from loss.loss_criterion import *
from validate import epoch_validate
from model.transformer import *

from eval.eval import EvaluationCriterion, Meter

import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
TQDM_BAR_FORMAT = '{desc} {n_fmt}/{total_fmt} [{elapsed} | {remaining} | {rate_fmt}]'

torch.manual_seed(1213)
np.random.seed(2022)
random.seed(1027)



def setup(rank, world_size, config):
	# Initialize the process group
	dist.init_process_group(
		backend=config['distributed']['backend'],
		init_method=config['distributed']['init_method'],
		rank=rank,
		world_size=world_size,
		timeout=datetime.timedelta(seconds=config['distributed']['timeout'])
	)
	# Set the GPU to use
	torch.cuda.set_device(rank)


def cleanup():

	dist.destroy_process_group()


def get_loader(dataset, batch_size,world_size, rank):
	sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
	data_loader = DataLoader(dataset,
				  batch_size=batch_size,
				  shuffle=False,
				  num_workers=6,
				  sampler=sampler, 
				  drop_last=False,              
				  collate_fn=dataset.collate_fn)
	return data_loader, sampler




def get_dataset(rank, world_size, dataroot, phase, batch_size, r, space):
	
	data = read_data(phase)
	dataset = create_dataloader(data,
								dataroot,
								batch_size, 
								rank=rank,                                   
								cache='ram', # if opt.cache == 'val' else opt.cache,
								workers=6,
								phase=phase,
								shuffle=True,
								r=r,
								space=space)
	
	return dataset



def compute_loss(outputs, targets, criterion, nc=2, wb=False):

	loss_dict = criterion(outputs, targets) #.to(outputs.device)
	if not loss_dict['loss_ce']:
		print(loss_dict)
	weight_dict = criterion.weight_dict

	loss_elements = [loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict]
	total_loss = sum(loss_elements)
	
	loss_elements = torch.stack(loss_elements)
	if wb:
		wandb.log({'train_'+a:b for a,b in zip(weight_dict.keys(), loss_elements)})

	return loss_elements, total_loss


def train_epoch(rank, model, optimizer, train_loader, epoch, epochs, criterion, nc, wb=False):
		
	model.train()
	criterion.train()
	ls = Meter(1, rank)
	ls_dict = Meter(3, rank)

	# training_criterion = EvaluationCriterion(rank, plot=False)
		

	if rank==0:
		print(('\n\n' + '%44s'+'%11s' * 6) % ('***Training***', 'Epoch', 'GPU Mem', 'ce_loss', 'bb_loss', 'iou_loss', 'mean_loss'))

	pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)

	for batch_idx, (img, target,fns) in pbar:
		img = img.to(rank, non_blocking=True)
		target = [t.to(rank, non_blocking=True) for t in target]
		optimizer.zero_grad()

		outputs = model(img.permute(1,0,2,3,4))
		outputs = {'pred_logits':outputs[0], 'pred_boxes': outputs[1]}
		targets = [{'labels': t[:,0], 'boxes':t[:,1:]} for t in target]
	
		l_dict, loss = compute_loss(outputs, targets, criterion, nc, wb)
		p = ls.adds(loss)
		d = ls_dict.adds(l_dict)

		loss.backward()
		optimizer.step()


		avg_ls = ls.returns(ls.means('r'))
		avg_ls_dict = ls_dict.returns(ls_dict.means())
		if wb:
			wandb.log({"train_loss": loss})
		

		if rank==0:
			mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.1g}G'  # (GB)
			pbar.set_description(('%44s'+'%11s'*2 + '%11.4s'*4) % 
								(f' ', f'{epoch}/{epochs - 1}', 
								mem,f'{avg_ls_dict[0]}', f'{avg_ls_dict[1]}', f'{avg_ls_dict[2]}', f'{avg_ls}'))
			
		# if batch_idx>2:
		# 	break

	return model, avg_ls
	
	
	
def run_eval(rank, root, epoch, lr_scheduler, model, val_loader, criterion_val, nc, best_fitness, config):
	
	fitness, valloss = epoch_validate(rank, model, val_loader, criterion_val, nc, wb=config['wandb']['enabled'])
	if config['wandb']['enabled']:
		wandb.log({'fitness':fitness})
	
	if rank==0:
		output_dir = root + config['paths']['output_dir']
		os.makedirs(output_dir, exist_ok=True)
		
		if fitness>best_fitness:
			save_path = f'{output_dir}detection_best.pth'
			best_fitness = fitness
		else:
			save_path = f'{output_dir}detection_last.pth'

		print(('\n%44s' + '%22s') % ('Saved model as:', save_path))
		checkpoint = {
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'lr_state_dict': lr_scheduler.state_dict(),
				'fitness': fitness,
				'best_fitness': best_fitness,
			}
		torch.save(checkpoint, save_path)
	return model, fitness, best_fitness, valloss


def load_saved_model(weights_path, root, M, O=None):
	# Handle both absolute paths and relative paths
	if os.path.isabs(weights_path) or os.path.exists(weights_path):
		ckptfile = weights_path
	elif os.path.exists(root + weights_path):
		ckptfile = root + weights_path
	else:
		ckptfile = root + 'runs/' + weights_path + '.pth'
	
	if not os.path.exists(ckptfile):
		raise FileNotFoundError(f"Checkpoint file not found: {ckptfile}")
		
	ckpts = torch.load(ckptfile, map_location='cpu')
	ckpt = ckpts['model_state_dict']
	
	if O is None:
		# Loading pretrained encoder weights
		new_state_dict = OrderedDict()
		for key, value in ckpt.items():
			new_key = key.replace('module.encoder.', '')
			new_state_dict[new_key] = value
		M.load_state_dict(new_state_dict, strict=False)
		print(f'Loaded pretrained encoder weights from {ckptfile}')
		if 'best_val_acc' in ckpts:
			print(f'Pretrained model best validation accuracy: {ckpts["best_val_acc"]:.4f}')

	
	if O is not None:
		# Resuming full training
		M.load_state_dict(ckpt)
		if 'lr_state_dict' in ckpts:
			O.load_state_dict(ckpts['lr_state_dict'])
		elif 'scheduler_state_dict' in ckpts:
			O.load_state_dict(ckpts['scheduler_state_dict'])
		start_epoch = ckpts['epoch']+1
		best_accuracy = ckpts.get('best_fitness', ckpts.get('fitness', 0))
		return M, O, start_epoch, best_accuracy

	return M
	


# def detector(rank, world_size, root, dataroot, pretraining=False, pretrained_weights_path='best_pretrainer.pth', resume=False):
def detector(rank, world_size, config):
	setup(rank, world_size, config)
	
	# Extract config parameters
	nc = config['model']['num_classes']
	epochs = config['training']['epochs']
	r = config['data']['r']
	space = config['data']['space']
	batch_size = config['training']['train_batch_size']
	val_batch_size = config['training']['val_batch_size']

	dataroot = config['paths']['dataroot']
	root = config['paths']['root']

	resume = config['resume']['enabled']
	resume_weights = config['resume']['weights_path']

	pretraining = config['pretrain']['use_pretrained']
	pretrain_weights = config['paths']['pretrain_weights']

	encoder = Encoder(
		hidden_dim=config['model']['hidden_dim'],
		num_encoder_layers=config['model']['num_encoder_layers'],
		nheads=config['model']['nheads']
	).to(rank)
	# encoder = DDP(encoder, device_ids=[rank], find_unused_parameters=False)
	for i, n in encoder.named_parameters():
		# if n.requires_grad:
		print(i)
	print('\n\n')
	if pretraining:
		encoder = load_saved_model(pretrain_weights, root, encoder, None)
		if config['pretrain']['freeze_encoder']:
			for param in encoder.parameters():
				param.requires_grad = False
			print('Pretrained model {} loaded (frozen)'.format(pretrain_weights))
		else:
			print('Pretrained model {} loaded'.format(pretrain_weights))

	train_data = get_dataset(rank, world_size, dataroot, 'train', batch_size, r, space)
	val_dataset = get_dataset(rank, world_size, dataroot, 'val', batch_size, r, space)
	# gc.collect()
	
	# define detection model
	model = Dent_Pt(
		encoder, 
		hidden_dim=config['model']['hidden_dim'], 
		num_class=config['model']['num_classes']
	).to(rank)
	model.train()
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	if rank == 0:
		print(f'Total trainable parameters: {pytorch_total_params:,}')
	model = DDP(model, device_ids=[rank], find_unused_parameters=True)

	# declare optimizer and scheduler
	optimizer = torch.optim.AdamW(
		model.parameters(), 
		lr=config['training']['learning_rate'],
		weight_decay=config['training']['weight_decay'],
		foreach=None, 
		fused=True
	)
	
	lr_config = config['training']['lr_scheduler']
	lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, 
		patience=lr_config['patience'],
		cooldown=lr_config['cooldown'],
		factor=lr_config['factor'],
		mode=lr_config['mode']
	)

	train_loader, sampler = get_loader(train_data, batch_size, world_size, rank)
	val_loader, _ = get_loader(val_dataset, val_batch_size, world_size, rank)

	
	# del train_data
	# del val_dataset
	torch.cuda.empty_cache()
	gc.collect()

	criterion_train,_ = loss_functions(nc, phase='train')
	criterion_val,_ = loss_functions(nc, phase='val')

	best_fitness = 0

	start_epoch = 0

	if resume:
		model, lr_scheduler, start_epoch, accuracy, best_accuracy = load_saved_model(resume_weights, root, model, optimizer)
		if rank == 0:
			print('Resuming training from epoch {}. Loaded weights from {}. Last best accuracy was {}'
				.format(start_epoch, resume_weights, best_accuracy))
	
	wb = config['wandb']['enabled']
	if wb:
		wandb.login()
		wandb.init(
			project=config['wandb']['project'], 
			name=config['wandb']['name'], 
			config={
				"architecture": "DENT",
				"dataset": "SCR",
				"epochs": epochs,
				**config
			})
  
	
	for epoch in range(start_epoch, epochs):
		sampler.set_epoch(epoch)
		lr = lr_scheduler.optimizer.param_groups[0]['lr']
		if wb:
			wandb.run.summary['LR'] = lr
			wandb.define_metric("loss", summary="min")
			wandb.define_metric("best_fitness", summary="max")

		model, train_loss = train_epoch(rank, model, optimizer, train_loader, epoch, epochs, criterion_train, nc=nc, wb=wb)		
		model, fitness, best_fitness, loss = run_eval(rank, root, epoch, lr_scheduler, model, val_loader, criterion_val, nc, best_fitness, config)
		lr_scheduler.step(train_loss)

	if wb:
		wandb.log_artifact(model)
		wandb.finish()	
	cleanup()
		


def load_config(config_path='config.yaml'):
	"""Load configuration from YAML file."""
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config


def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')
	parser.add_argument('--world_size', type=int, default=None, help='override world size from config')
	parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
	parser.add_argument('--batch_size', type=int, default=None, help='override batch size')
	parser.add_argument('--lr', type=float, default=None, help='override learning rate')
	return parser.parse_args()


if __name__ == '__main__':
	args = arg_parse()
	
	# Load config from YAML
	config = load_config(args.config)
	
	# Override config with command line arguments if provided
	if args.world_size is not None:
		config['distributed']['world_size'] = args.world_size
	if args.epochs is not None:
		config['training']['epochs'] = args.epochs
	if args.batch_size is not None:
		config['training']['train_batch_size'] = args.batch_size
		config['training']['val_batch_size'] = args.batch_size
	if args.lr is not None:
		config['training']['learning_rate'] = args.lr
	
	world_size = config['distributed']['world_size']
	mp.spawn(detector, args=(world_size, config), nprocs=world_size, join=True)
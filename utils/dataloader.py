"""
Dataloaders and dataset utils
"""

import contextlib
import glob
import pickle
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import Image, ImageOps
from torchvision.ops import nms
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

# from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
#                                  cutout, letterbox, mixup, random_perspective)
from utils.general import (check_dataset, check_requirements, check_yaml, clean_str,
						   colorstr, cv2, unzip_file, xyn2xy, xywh2xyxy,
						   xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first
from utils.util import root_tail


# random.seed(888100)
torch.manual_seed(1213)
np.random.seed(2022)
random.seed(1027)

# Parameters
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def create_dataloader(data_list,
					dataroot,
					batch_size,
					rank,
					augment=False,
					cache=False,
					workers=8,
					phase='val',
					shuffle=False,
					r=3,
					space=1):
	
	
	with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
		dataset = LoadImagesAndLabels(rank,
			data_list,
			dataroot,
			batch_size,
			augment=augment,  
			cache_images=cache,
			phase=phase,
			r=r,
			space=space)
	return dataset


class LoadImagesAndLabels(Dataset):

	def __init__(self,
				rank,
				data_list,
				dataroot, 
				batch_size=16,
				augment=False,
				cache_images=False,
				phase='val',
				r=3,
				space=1):

		self.augment = augment
		self.dataroot = dataroot
		self.phase = phase
		self.r = r
		self.rank = rank
		self.space = space


		try:
			
			self.im_files = data_list
			random.shuffle(self.im_files)
			assert self.im_files, f'No images found'
		except Exception as e:
			raise Exception(f'Error loading data from {path}: {e}\n{HELP_URL}') from e


		n = len(self.im_files)  # number of images
		self.n = n
		self.indices = range(n)


		par = Path(self.dataroot)
		sar = self.phase+'filename'
		car = self.phase+'index'
		filecache_path = (par/sar).with_suffix('.cache')
		indexcache_path = (par/car).with_suffix('.cache')
		c_file, c_index = self.cache_files(filecache_path, indexcache_path)

		self.roots = c_file
		self.c_index = c_index

	def cache_files(self, p, q):
		if p.is_file() and q.is_file():
			files= np.load(p, allow_pickle=True).item()
			indexfile= np.load(q, allow_pickle=True).item()
			return files, indexfile

		files = {}
		roots = []
		indexfile = {}
		print('\nCreating index to root caches since they dont exist (yet)...\n')
		for i in range(len(self.im_files)):
			f = self.im_files[i]
			root, tail = self.get_roots(f)
			indexfile[f] = [i, root, tail]

			if root not in roots:
				roots.append(root)
				files[root] = [tail]
			else:
				files[root].append(tail)
				files[root] = sorted(files[root])

		# p = Path(self.dataroot) / p 
		# q = Path(self.dataroot) / q

		if self.rank==0:

			np.save(p, files)
			p.with_suffix('.cache.npy').rename(p)

			np.save(q, indexfile)
			q.with_suffix('.cache.npy').rename(q)

		return files, indexfile


	def get_roots(self, file):
		# name = (file.split('/')[-1]).split('.')
		# ext = name[-1].split('.')[0]
		root, tail = root_tail(file)
		# root = '../'+root
		
		return root, int(tail)


	def __len__(self):
		return len(self.im_files)


		# *************************************************************************************************

	def create_labelout(self, b, c):
		# [bi, class, x, y, w, h]
		# Filter out label 2 (no boxes) - keep only 0 (Fovea) and 1 (SCR)
		# Convert to tensors if not already
		if isinstance(c, torch.Tensor):
			c = c.cpu().numpy() if c.is_cuda else c.numpy()
		if isinstance(b, torch.Tensor):
			b = b.cpu().numpy() if b.is_cuda else b.numpy()
		
		c = np.array(c).flatten()
		b = np.array(b).reshape(-1, 4)
		
		# Filter out invalid boxes (label == 2 or all-zero boxes)
		valid_mask = (c != 2) & ~np.all(b == 0, axis=1)
		c_filtered = c[valid_mask]
		b_filtered = b[valid_mask]
		
		if len(c_filtered) == 0:
			# No valid boxes - return empty tensors
			b_tensor = torch.empty((0, 4))
			c_tensor = torch.zeros((0, 1))
		else:
			# Add 1 to labels: 0→1 (Fovea becomes class 1), 1→2 (SCR becomes class 2)
			c_filtered = c_filtered + 1
			b_tensor = torch.from_numpy(b_filtered).float()
			c_tensor = torch.from_numpy(c_filtered).reshape((-1, 1)).float()

		d = torch.zeros_like(c_tensor)
		lab = torch.cat([d, c_tensor, b_tensor], dim=-1).float()
		return lab



	def lookup(self, index):
		if self.r == 1:
			return [index]
		file = self.im_files[index] # index filename
		i, root, tail = self.c_index[file] # filename [index, root, tail]
		a = np.array(self.roots[root]) # root [tail1, tail2, ....]
		n1 = np.where(a==tail)[0][0] # tail index

		n0 = n1 - self.space
		n2 = n1 + self.space

		neighborhood = [0,index,0]

		if n0<0:
			neighborhood[0] = index
		else:
			filename = root + '_' + str(a[n0]) + '.pkl'
			
			n0_ind = self.c_index[filename][0]
			neighborhood[0] = n0_ind

		if n2>=len(a):
			neighborhood[2] = index
		else:
			filename = root + '_' + str(a[n2]) + '.pkl'
			n2_ind = self.c_index[filename][0]
			neighborhood[2] = n2_ind

		return neighborhood




	def load_pickle(self, index):
		im_file = self.im_files[index]
		with open(im_file, 'rb') as handle:
			d = pickle.load(handle)
			im = d['img']
			bo = d['box']
			clss = d['label']
			nm = d['name']
			
			# Ensure numpy arrays for consistent processing
			if isinstance(im, torch.Tensor):
				im = im.numpy()
			if isinstance(bo, torch.Tensor):
				bo = bo.numpy()
			if isinstance(clss, torch.Tensor):
				clss = clss.numpy()
				
		return im, bo, clss, nm



	def __getitem__(self, index):
		r = self.r
		space = self.space

		neighborhood = self.lookup(index)

		image = [None] * r
		labels = []
		nms = ''
		for i, ind in enumerate(neighborhood):
			img, box, clss, nm = self.load_pickle(ind)
			# Images are stored as [C, H, W] tensors, keep as is
			image[i] = img
			if ind == index:
				labels = self.create_labelout(box, clss)
				nms = nm
			 
		# Stack images: shape will be [r, C, H, W]
		image = np.stack(image)

		image = torch.from_numpy(np.ascontiguousarray(image).astype(np.float32))
		return image, labels, nms


	@staticmethod
	def collate_fn(batch):
		im, label, name = zip(*batch)  # transposed        
	   
		label = [l[:,1:] for l in label]
		im = torch.stack(im, 0)
		name = [nm for nm in name]

		return im, label, name





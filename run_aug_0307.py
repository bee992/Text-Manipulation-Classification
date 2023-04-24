from logging import Logger
import os
from threading import Timer
from lookahead import Lookahead
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from sklearn import metrics
from torch.utils.data import RandomSampler
from model import *
from dataset import *

import torch.cuda.amp as amp
is_amp = True  #True #False
import torch 
from torch.nn.parallel.data_parallel import data_parallel
from matplotlib import pyplot as plt
from utils import *
import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold, KFold # Sklearn

#################################################################################################
torch.backends.cudnn.enabled = False


def do_valid(net, valid_loader):

	valid_num = 0
	valid_loss = 0
	valid_probability = []
	valid_truth = []

	net = net.eval()
	start_timer = time.time()
	for t, batch in enumerate(tqdm(valid_loader)):
		
		net.output_type = ['loss', 'inference']
		with torch.no_grad():
			with amp.autocast(enabled = is_amp):
				
				batch_size = len(batch['index'])
				for k in ['image', 'label' ]: batch[k] = batch[k].cuda()
				
				output = data_parallel(net, batch) #net(input)#
				loss0  = output['bce_loss'].mean()
				# loss1  = output['focal_loss'].mean()



		valid_num += 1
		valid_loss += loss0.item()
		valid_truth.append(batch['label'].data.cpu().numpy())
		valid_probability.append(output['label'].data.cpu().numpy())

	truth = np.concatenate(valid_truth)
	probability = np.concatenate(valid_probability)

	pred_tampers = probability[truth == 1]
	pred_untampers = probability[truth == 0]

	thres = np.percentile(pred_untampers, np.arange(90, 100, 1))
	recall = np.mean(np.greater(pred_tampers[:, np.newaxis], thres).mean(axis=0))
	#------
	loss = valid_loss/valid_num
	score = recall * 100
	pf1 = pfbeta(truth, probability, beta=1)
	AUC = metrics.roc_auc_score(truth, probability)
	return [loss, pf1, AUC, score, thres[0]]



 ##----------------

def run_train():

	tampered_img_paths = "../tianchi_data/data/train/tampered/imgs"
	untampered_img_paths = "../tianchi_data/data/train/untampered/"

	col_name = ['img_name', 'img_path', 'img_label']
	imgs_info = [] 
	for img_name in os.listdir(tampered_img_paths):
		if img_name.endswith('.jpg'):
			imgs_info.append(["p_"+img_name, os.path.join(tampered_img_paths, img_name), np.ones(1).astype(np.float32)])
		
	for img_name in os.listdir(untampered_img_paths):
		if img_name.endswith('.jpg'): 
			imgs_info.append(["n_"+img_name, os.path.join(untampered_img_paths, img_name), np.zeros(1).astype(np.float32)])
	
	imgs_info_array = np.array(imgs_info)    
	df = pd.DataFrame(imgs_info_array, columns=col_name)


	root_dir = '../user_data'
	out_dir  = root_dir + '/result/0307/effb6-1024-v00a-RandomSampler-StepLR'
	initial_checkpoint = None


	n_fold = 5
	kf = KFold(n_splits=n_fold, shuffle=True, random_state=4307)
	for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
		df.loc[val_idx, 'fold'] = fold

	
	for fold in range(n_fold):
		fold_dir = out_dir  + f'/fold-{fold}'

		print(f'#'*40, flush=True)
		print(f'###### Fold: {fold}', flush=True)
		print(f'#'*40, flush=True)

		train_df, valid_df = make_fold(df, fold)
		train_dataset = DTTDataset(train_df, augment=train_augment_v00a)
		valid_dataset = DTTDataset(valid_df, augment=None)
		
		# pdb.set_trace()
		start_lr   = 5e-4 #0.0001
		batch_size = 32 # bs32 for A100_80G

		train_loader  = DataLoader(train_dataset,
		sampler = RandomSampler(train_dataset),
		batch_size  = batch_size,
		drop_last   = False,
		num_workers = 8,
		pin_memory  = False,
		worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
		collate_fn = null_collate,
		)
 
		valid_loader = DataLoader(valid_dataset,
		sampler = SequentialSampler(valid_dataset),
		batch_size  = batch_size * 2,
		drop_last   = False,
		num_workers = 8,
		pin_memory  = False,
		collate_fn = null_collate,
		)

		## setup  ----------------------------------------
		for f in ['checkpoint','train','valid','backup'] : os.makedirs(fold_dir +'/'+f, exist_ok=True)
		
		log = open(fold_dir+'/log_train.txt',mode='a')
		
		log.write(f'\tfold_dir = {fold_dir}\n' )
		log.write('\n')


		## dataset ----------------------------------------
		log.write('** dataset setting **\n')
		log.write(f'fold = {fold}\n')
		log.write(f'train_dataset : \n{train_dataset}\n')
		log.write(f'valid_dataset : \n{valid_dataset}\n')
		log.write('\n')

		## net ----------------------------------------
		log.write('** net setting **\n')
		
		scaler = amp.GradScaler(enabled = is_amp)
		net = Net().cuda()

		if initial_checkpoint is not None:
			f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
			start_iteration = f['iteration']
			start_epoch = f['epoch']
			# start_iteration = 0
			# start_epoch = 0
			state_dict  = f['state_dict']
			net.load_state_dict(state_dict,strict=False)  #True
		else:
			start_iteration = 0
			start_epoch = 0
			net.load_pretrain()


		log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
		log.write('\n')


		# optimizer = Lookahead(torch.optim.RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr))
		optimizer = torch.optim.RAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr, betas=(0.9, 0.999))
		scheduler = get_scheduler(optimizer)

		# optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr), alpha=0.5, k=5)
		
		log.write('optimizer\n  %s\n'%(optimizer))
		log.write('\n')
		
		num_iteration = 20*len(train_loader)
		iter_log   = int(len(train_loader)*1) #479
		iter_valid = iter_log
		iter_save  = iter_log
	
		## start training here! ##############################################
		log.write('** start training here! **\n')
		log.write('   batch_size = %d \n'%(batch_size))
		log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
		log.write('                     |-------------- VALID---------|---- TRAIN/BATCH ----------------\n')
		log.write('rate     iter  epoch | loss  pF1  AUC  score  thres | loss           | time           \n')
		log.write('-------------------------------------------------------------------------------------\n')
		
		def message(mode='print'):
			asterisk = ' '
			if mode==('print'):
				loss = batch_loss
			if mode==('log'):
				loss = train_loss
				if (iteration % iter_save == 0): asterisk = '*'
			
			text = \
				('%0.2e   %08d%s %6.2f | '%(rate, iteration, asterisk, epoch,)).replace('e-0','e-').replace('e+0','e+') + \
				'%4.4f  %4.4f  %4.4f  %4.4f  %4.4f | '%(*valid_loss,) + \
				'%4.4f  %4.4f  %4.3f  | ' % (*loss,) + \
				'%s' % (time_to_str(time.time() - start_timer,'min'))
			
			return text

		#----
		valid_loss = np.zeros(5,np.float32)

		train_loss = np.zeros(3,np.float32)
		batch_loss = np.zeros_like(train_loss)
		sum_train_loss = np.zeros_like(train_loss)
		sum_train = 0
		
		start_timer = time.time()
		iteration = start_iteration
		epoch = start_epoch
		rate = 0
		while iteration < num_iteration:
			for t, batch in enumerate((train_loader)):

				if iteration%iter_save==0 and iteration != start_iteration:
						torch.save({
							'state_dict': net.state_dict(),
							'iteration': iteration,
							'epoch': epoch,
						}, f'{fold_dir}/checkpoint/{epoch}.model.pth')
						pass
				
				
				if (iteration > 0 and iteration%iter_valid==0): 
					valid_loss = do_valid(net, valid_loader)
					pass
				
				
				if (iteration%iter_log==0) or (iteration%iter_valid==0):
					print('\r', end='', flush=True)
					log.write(message(mode='log') + '\n')
					
					
				
				rate = get_learning_rate(optimizer) 
				
				# one iteration update  -------------
				batch_size = len(batch['index'])
				for k in ['image', 'label' ]: batch[k] = batch[k].cuda()

				net.train()
				net.output_type = ['loss', 'inference']

				if 1:
					with amp.autocast(enabled = is_amp):
						output = data_parallel(net,batch)
						loss0  = output['bce_loss'].mean()
						# loss1  = output['focal_loss'].mean()


					optimizer.zero_grad()
					scaler.scale(loss0).backward()
					
					scaler.unscale_(optimizer)
					scaler.step(optimizer)
					scaler.update()
				
				
				# print statistics  --------
				batch_loss[:3] = [loss0.item(), 0, 0]
				sum_train_loss += batch_loss
				sum_train += 1
				if t % 100 == 0:
					train_loss = sum_train_loss / (sum_train + 1e-12)
					sum_train_loss[...] = 0
					sum_train = 0
				
				print('\r', end='', flush=True)
				print(message(mode='print'), end='', flush=True)
				epoch += 1 / len(train_loader)
				iteration += 1
				
				# debug  --------
				if 0:
					image = batch['image'].float().data.cpu().numpy()
					truth = batch['mask'].long().data.cpu().numpy()
					predict = output['mask'].float().data.cpu().numpy()

					image = np.ascontiguousarray(image.transpose(0,2,3,1))
					predict = np.ascontiguousarray(predict.transpose(0,2,3,1))
					for b in range(batch_size):
						m = image[b]
						p = predict[b]
						t = truth[b,0]
						onehot = np.ascontiguousarray(np.eye(4)[t][...,1:])
						p = np.ascontiguousarray(p[..., 1:])

			scheduler.step()
			
					
			torch.cuda.empty_cache()
		log.write('\n')

# main #
if __name__ == '__main__':
	run_train()

'''
 

'''

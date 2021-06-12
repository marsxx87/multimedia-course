import argparse
import gc
import logging
import os
import sys
import time
import math
import numpy as np

import datetime

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss, direction_loss
from sgan.losses import directional_error, final_directional_error, displacement_error, final_displacement_error

from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path
from sgan.sceneseg import SceneModule
from sgan.directionmod import (DirectionGenerator, DirectionDiscriminator, 
      direction_discriminator_step, direction_generator_step, obs_to_direction, pred_to_direction)

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

# Dataset options
parser.add_argument('--dataset_name', default='04171525', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--coord_scale', default=100 , type=int)

# Optimization
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=500, type=int)
parser.add_argument('--num_epochs', default=500, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=64, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=64, type=int)
parser.add_argument('--noise_dim', default=(8, ), type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped') #'ped' or 'global'
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--g_pool_net_on', default=1, type=bool_flag) #P for generator
parser.add_argument('--g_direction_on', default=0, type=bool_flag) #D for generator
parser.add_argument('--g_typeID_on', default=0, type=bool_flag) #Ty for generator
parser.add_argument('--g_scene_on', default=0, type=bool_flag) #S for generator
parser.add_argument('--bottleneck_dim', default=1024, type=int)
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Direction SGAN Option
parser.add_argument('--dir_num_epochs', default=40, type=int)
parser.add_argument('--dir_g_learning_rate', default=1e-3, type=float)
parser.add_argument('--dir_d_learning_rate', default=1e-2, type=float)
parser.add_argument('--dir_output_dir', default=os.getcwd())
parser.add_argument('--dir_checkpoint_every', default=500, type=int)
parser.add_argument('--dir_checkpoint_name', default='dir_Yangde')
parser.add_argument('--dir_checkpoint_start_from', default=None)
parser.add_argument('--dir_restore_from_checkpoint', default=1, type=int)
parser.add_argument('--dir_new_start', default=1, type=bool_flag)
parser.add_argument('--dir_train', default=0, type=bool_flag)

# Discriminator Options
parser.add_argument('--d_type', default='global', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Scene Segmentation Options
parser.add_argument('--seg_scene_width', default=30, type=int) #width of the rectangle used in scene module
parser.add_argument('--seg_scene_height', default=15, type=int) #height of the rectangle used in scene module
parser.add_argument('--seg_scene_file', default='segmentation/Yangde.png') #filename of segmented scene
parser.add_argument('--seg_scale', default=10, type=int) #scale of segmentation
parser.add_argument('--scene_object_types', default=4, type=int) #scene_object_types for scene module

# Loss Options
parser.add_argument('--l2_loss_weight', default=1.0, type=float)
parser.add_argument('--direction_loss_weight', default=1.0, type=float)
parser.add_argument('--best_k', default=20, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=20, type=int)
parser.add_argument('--checkpoint_every', default=20, type=int)
parser.add_argument('--checkpoint_name', default='SGAN_04171525')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)
parser.add_argument('--new_start', default=1, type=bool_flag)
parser.add_argument('--sgan_train', default=1, type=bool_flag)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)
parser.add_argument('--logname', default="log.log", type=str)

parser.add_argument('--use_gru', default=0, type=int)

args = parser.parse_args()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

if args.logname:
      fh = logging.FileHandler(args.logname)
      fh.setLevel(logging.WARNING)
      logger.addHandler(fh)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\n"):
      """
      Call in a loop to create terminal progress bar
      @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
      """
      percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
      filledLength = int(length * iteration // total)
      bar = fill * filledLength + '-' * (length - filledLength)
      print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
      # Print New Line on Complete
      # if iteration == total: 
      #       print()
            
def init_weights(m):
      classname = m.__class__.__name__
      if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight)

def get_dtypes(args):
      long_dtype = torch.LongTensor
      float_dtype = torch.FloatTensor
      if args.use_gpu == 1:
            long_dtype = torch.cuda.LongTensor
            float_dtype = torch.cuda.FloatTensor
      return long_dtype, float_dtype

def main(args):
      os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
      train_path = get_dset_path(args.dataset_name, 'train')
      val_path = get_dset_path(args.dataset_name, 'val')

      long_dtype, float_dtype = get_dtypes(args)

      logger.warning("Initializing train dataset: {}".format(args.dataset_name))
      train_dset, train_loader = data_loader(args, train_path)
      logger.info("Initializing val dataset")
      _, val_loader = data_loader(args, val_path)

      if args.dir_train:
            iterations_per_epoch = len(train_dset) / args.batch_size
            if args.dir_num_epochs:
                  args.num_iterations = int(iterations_per_epoch * args.dir_num_epochs)
            
            logger.info('There are {} iterations in total, {} iterations per epoch'.format(args.num_iterations, iterations_per_epoch))

            dir_generator = DirectionGenerator(
                  obs_len=args.obs_len, 
                  pred_len=args.pred_len, 
                  embedding_dim=args.embedding_dim, 
                  encoder_h_dim=args.encoder_h_dim_g, 
                  decoder_h_dim=args.decoder_h_dim_g, 
                  mlp_dim=args.mlp_dim, 
                  num_layers=args.num_layers, 
                  noise_dim=args.noise_dim, 
                  noise_type=args.noise_type, 
                  noise_mix_type=args.noise_mix_type, 
                  dropout=args.dropout, 
                  bottleneck_dim=args.bottleneck_dim, 
                  batch_norm=args.batch_norm, 
            )
            
            dir_generator.apply(init_weights)
            dir_generator.type(float_dtype).train()
            # logger.info('Here is the direction generator:')
            # logger.info(dir_generator)

            dir_discriminator = DirectionDiscriminator(
                  obs_len=args.obs_len, 
                  pred_len=args.pred_len, 
                  embedding_dim=args.embedding_dim, 
                  h_dim=args.encoder_h_dim_d, 
                  mlp_dim=args.mlp_dim, 
                  num_layers=args.num_layers, 
                  dropout=args.dropout, 
                  batch_norm=args.batch_norm
            )

            dir_discriminator.apply(init_weights)
            dir_discriminator.type(float_dtype).train()
            # logger.info('Here is the direction discriminator:')
            # logger.info(dir_discriminator)

            dir_g_loss_fn = gan_g_loss
            dir_d_loss_fn = gan_d_loss

            optimizer_dir_g = optim.Adam(dir_generator.parameters(), lr=args.dir_g_learning_rate)
            optimizer_dir_d = optim.Adam(dir_discriminator.parameters(), lr=args.dir_d_learning_rate)

            dir_restore_path = None
            if args.dir_checkpoint_start_from is not None:
                  dir_restore_path = args.dir_checkpoint_start_from
            elif args.dir_restore_from_checkpoint == 1:
                  dir_restore_path = os.path.join(args.dir_output_dir, '%s_with_model.pt' % args.dir_checkpoint_name)

            if dir_restore_path is not None and os.path.isfile(dir_restore_path):
                  logger.info('Restoring from checkpoint {}'.format(dir_restore_path))
                  checkpoint = torch.load(dir_restore_path)
                  dir_generator.load_state_dict(checkpoint['g_state'])
                  dir_discriminator.load_state_dict(checkpoint['d_state'])
                  optimizer_dir_g.load_state_dict(checkpoint['g_optim_state'])
                  optimizer_dir_d.load_state_dict(checkpoint['d_optim_state'])
                  dir_t = checkpoint['counters']['t']
                  dir_epoch = checkpoint['counters']['epoch']
                  checkpoint['restore_ts'].append(dir_t)
                  if args.dir_new_start:
                        dir_t, dir_epoch = 0, 0
            else:
                  # Starting from scratch, so initialize checkpoint data structure
                  dir_t, dir_epoch = 0, 0
                  checkpoint = {
                        'args': args.__dict__, 
                        'G_losses': defaultdict(list), 
                        'D_losses': defaultdict(list), 
                        'losses_ts': [], 
                        'metrics_val': defaultdict(list), 
                        'metrics_train': defaultdict(list), 
                        'sample_ts': [], 
                        'restore_ts': [], 
                        'norm_g': [], 
                        'norm_d': [], 
                        'counters': {
                              't': None, 
                              'epoch': None, 
                        }, 
                        'g_state': None, 
                        'g_optim_state': None, 
                        'd_state': None, 
                        'd_optim_state': None, 
                        'g_best_state': None, 
                        'd_best_state': None, 
                        'best_t': None, 
                        'g_best_nl_state': None, 
                        'd_best_state_nl': None, 
                        'best_t_nl': None, 
                  }

            def dir_saveCheckpoint():
                  checkpoint['counters']['t'] = dir_t
                  checkpoint['counters']['epoch'] = dir_epoch
                  checkpoint['sample_ts'].append(dir_t)

                  # Check stats on the validation set
                  logger.info('Checking stats on val ...')
                  metrics_val = dir_check_accuracy(
                        args, val_loader, dir_generator, dir_discriminator, dir_d_loss_fn
                  )
                  logger.info('Checking stats on train ...')
                  metrics_train = dir_check_accuracy(
                        args, train_loader, dir_generator, dir_discriminator,
                        dir_d_loss_fn, limit=True
                  )

                  for k, v in sorted(metrics_val.items()):
                        logger.info('  [val] {}: {:.3f}'.format(k, v))
                        checkpoint['metrics_val'][k].append(v)
                  for k, v in sorted(metrics_train.items()):
                        logger.info('  [train] {}: {:.3f}'.format(k, v))
                        checkpoint['metrics_train'][k].append(v)

                  min_ade = min(checkpoint['metrics_val']['ade'])
                  min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                  if metrics_val['ade'] == min_ade:
                        logger.info('New low for avg_disp_error')
                        checkpoint['best_t'] = dir_t
                        checkpoint['g_best_state'] = dir_generator.state_dict()
                        checkpoint['d_best_state'] = dir_discriminator.state_dict()

                  if metrics_val['ade_nl'] == min_ade_nl:
                        logger.info('New low for avg_disp_error_nl')
                        checkpoint['best_t_nl'] = dir_t
                        checkpoint['g_best_nl_state'] = dir_generator.state_dict()
                        checkpoint['d_best_nl_state'] = dir_discriminator.state_dict()

                  # Save another checkpoint with model weights and
                  # optimizer state
                  checkpoint['g_state'] = dir_generator.state_dict()
                  checkpoint['g_optim_state'] = optimizer_dir_g.state_dict()
                  checkpoint['d_state'] = dir_discriminator.state_dict()
                  checkpoint['d_optim_state'] = optimizer_dir_d.state_dict()
                  dir_checkpoint_path = os.path.join(
                        args.dir_output_dir, '%s_with_model.pt' % args.dir_checkpoint_name
                  )
                  logger.info('Saving checkpoint to {}'.format(dir_checkpoint_path))
                  torch.save(checkpoint, dir_checkpoint_path)
                  logger.info('Saving process is done.')

            while dir_t < args.num_iterations:
                  gc.collect()
                  dir_d_steps_left = args.d_steps
                  dir_g_steps_left = args.g_steps
                  dir_epoch += 1
                  logger.info('Starting direction GAN epoch {}, {}'.format(dir_epoch, datetime.datetime.now()))
                  # printProgressBar(dir_t + 1, args.num_iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)

                  for batch in train_loader:

                        # Decide whether to use the batch for stepping on discriminator or
                        # generator; an iteration consists of args.d_steps steps on the
                        # discriminator followed by args.g_steps steps on the generator.
                        if dir_d_steps_left > 0:
                              step_type = 'd'
                              losses_d = direction_discriminator_step(args, batch, dir_generator, 
                                                            dir_discriminator, dir_d_loss_fn, 
                                                            optimizer_dir_d)
                              checkpoint['norm_d'].append(
                                    get_total_norm(dir_discriminator.parameters()))
                              dir_d_steps_left -= 1
                        elif dir_g_steps_left > 0:
                              step_type = 'g'
                              losses_g = direction_generator_step(args, batch, dir_generator, 
                                                            dir_discriminator, dir_g_loss_fn, 
                                                            optimizer_dir_g)
                              checkpoint['norm_g'].append(
                                    get_total_norm(dir_generator.parameters())
                              )
                              dir_g_steps_left -= 1

                        # Skip the rest if we are not at the end of an iteration
                        if dir_d_steps_left > 0 or dir_g_steps_left > 0:
                              continue
                  
                        # Maybe save loss
                        if dir_t % args.print_every == 0:
                              logger.info('t = {} / {}'.format(dir_t + 1, args.num_iterations))
                              for k, v in sorted(losses_d.items()):
                                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                                    checkpoint['D_losses'][k].append(v)
                              for k, v in sorted(losses_g.items()):
                                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                                    checkpoint['G_losses'][k].append(v)
                              checkpoint['losses_ts'].append(dir_t)
                        
                        # Maybe save a checkpoint                  
                        if (dir_t > 0 and dir_t % args.dir_checkpoint_every == 0) or dir_epoch >= args.dir_num_epochs*3:
                              dir_saveCheckpoint()

                        dir_t += 1
                        dir_d_steps_left = args.d_steps
                        dir_g_steps_left = args.g_steps
                        if dir_t >= args.num_iterations:
                              dir_saveCheckpoint()
                              logger.info("Training process is done.")
                              break
                  if dir_epoch % args.dir_num_epochs == 0:
                        dir_saveCheckpoint()
                        logger.info("Training process is done.")
                        break
      
      if args.sgan_train:
            iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
            if args.num_epochs:
                  args.num_iterations = int(iterations_per_epoch * args.num_epochs)
            
            args.seg_rec_size = np.array([args.seg_scene_width, args.seg_scene_height])

            logger.info('There are {} iterations in total, {} iterations per epoch'.format(args.num_iterations, iterations_per_epoch))

            if iterations_per_epoch < 1:
                  logger.warning("Iterations per epoch shouldn't be less than 1. Please set a lower batch size and start another training process again!")
                  pass

            generator = TrajectoryGenerator(
                  obs_len=args.obs_len, 
                  pred_len=args.pred_len, 
                  embedding_dim=args.embedding_dim, 
                  encoder_h_dim=args.encoder_h_dim_g, 
                  decoder_h_dim=args.decoder_h_dim_g, 
                  mlp_dim=args.mlp_dim, 
                  num_layers=args.num_layers, 
                  noise_dim=args.noise_dim, 
                  noise_type=args.noise_type, 
                  noise_mix_type=args.noise_mix_type, 
                  dropout=args.dropout, 
                  bottleneck_dim=args.bottleneck_dim, 
                  batch_norm=args.batch_norm, 
                  args=args 
            )
                  
            generator.apply(init_weights)
            generator.type(float_dtype).train()
            # logger.info('Here is the generator:')
            # logger.info(generator)
            
            discriminator = TrajectoryDiscriminator(
                  obs_len=args.obs_len, 
                  pred_len=args.pred_len, 
                  embedding_dim=args.embedding_dim, 
                  h_dim=args.encoder_h_dim_d, 
                  mlp_dim=args.mlp_dim, 
                  num_layers=args.num_layers, 
                  dropout=args.dropout, 
                  batch_norm=args.batch_norm, 
                  d_type=args.d_type,
                  args=args
            )

            discriminator.apply(init_weights)
            discriminator.type(float_dtype).train()
            # logger.info('Here is the discriminator:')
            # logger.info(discriminator)

            g_loss_fn = gan_g_loss
            d_loss_fn = gan_d_loss

            optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
            optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

            # Maybe restore from checkpoint
            restore_path = None
            if args.checkpoint_start_from is not None:
                  restore_path = args.checkpoint_start_from
            elif args.restore_from_checkpoint == 1:
                  restore_path = os.path.join(args.output_dir, '%s_with_model.pt' % args.checkpoint_name)

            if restore_path is not None and os.path.isfile(restore_path):
                  logger.info('Restoring from checkpoint {}'.format(restore_path))
                  checkpoint = torch.load(restore_path)
                  generator.load_state_dict(checkpoint['g_state'])
                  discriminator.load_state_dict(checkpoint['d_state'])
                  optimizer_g.load_state_dict(checkpoint['g_optim_state'])
                  optimizer_d.load_state_dict(checkpoint['d_optim_state'])
                  t = checkpoint['counters']['t']
                  epoch = checkpoint['counters']['epoch']
                  checkpoint['restore_ts'].append(t)
                  if args.new_start:
                        t, epoch = 0, 0      
            else:
                  # Starting from scratch, so initialize checkpoint data structure
                  t, epoch = 0, 0
                  checkpoint = {
                        'args': args.__dict__, 
                        'G_losses': defaultdict(list), 
                        'D_losses': defaultdict(list), 
                        'losses_ts': [], 
                        'metrics_val': defaultdict(list), 
                        'metrics_train': defaultdict(list), 
                        'sample_ts': [], 
                        'restore_ts': [], 
                        'norm_g': [], 
                        'norm_d': [], 
                        'counters': {
                              't': None, 
                              'epoch': None, 
                        }, 
                        'g_state': None, 
                        'g_optim_state': None, 
                        'd_state': None, 
                        'd_optim_state': None, 
                        'g_best_state': None, 
                        'd_best_state': None, 
                        'best_t': None, 
                        'g_best_nl_state': None, 
                        'd_best_state_nl': None, 
                        'best_t_nl': None, 
                  }

            def saveCheckpoint():
                  checkpoint['counters']['t'] = t
                  checkpoint['counters']['epoch'] = epoch
                  checkpoint['sample_ts'].append(t)

                  # Check stats on the validation set
                  logger.info('Checking stats on val ...')
                  metrics_val = check_accuracy(
                        args, val_loader, generator, discriminator, d_loss_fn
                  )
                  logger.info('Checking stats on train ...')
                  metrics_train = check_accuracy(
                        args, train_loader, generator, discriminator,
                        d_loss_fn, limit=True
                  )

                  for k, v in sorted(metrics_val.items()):
                        logger.info('  [val] {}: {:.3f}'.format(k, v))
                        checkpoint['metrics_val'][k].append(v)
                  for k, v in sorted(metrics_train.items()):
                        logger.info('  [train] {}: {:.3f}'.format(k, v))
                        checkpoint['metrics_train'][k].append(v)

                  min_ade = min(checkpoint['metrics_val']['ade'])
                  min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                  if metrics_val['ade'] == min_ade:
                        logger.info('New low for avg_disp_error')
                        checkpoint['best_t'] = t
                        checkpoint['g_best_state'] = generator.state_dict()
                        checkpoint['d_best_state'] = discriminator.state_dict()

                  if metrics_val['ade_nl'] == min_ade_nl:
                        logger.info('New low for avg_disp_error_nl')
                        checkpoint['best_t_nl'] = t
                        checkpoint['g_best_nl_state'] = generator.state_dict()
                        checkpoint['d_best_nl_state'] = discriminator.state_dict()

                  # Save another checkpoint with model weights and
                  # optimizer state
                  checkpoint['g_state'] = generator.state_dict()
                  checkpoint['g_optim_state'] = optimizer_g.state_dict()
                  checkpoint['d_state'] = discriminator.state_dict()
                  checkpoint['d_optim_state'] = optimizer_d.state_dict()
                  checkpoint_path = os.path.join(
                        args.output_dir, '%s_with_model.pt' % args.checkpoint_name
                  )
                  logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                  torch.save(checkpoint, checkpoint_path)
                  logger.info('Saving process is done.')

            while t < args.num_iterations:
                  gc.collect()
                  d_steps_left = args.d_steps
                  g_steps_left = args.g_steps
                  epoch += 1
                  logger.info('Starting epoch {}, {}'.format(epoch, datetime.datetime.now()))
                  # printProgressBar(t + 1, args.num_iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)

                  for batch in train_loader:
                        # Decide whether to use the batch for stepping on discriminator or
                        # generator; an iteration consists of args.d_steps steps on the
                        # discriminator followed by args.g_steps steps on the generator.
                        if d_steps_left > 0:
                              step_type = 'd'
                              losses_d = discriminator_step(args, batch, generator, 
                                                            discriminator, d_loss_fn, 
                                                            optimizer_d)
                              checkpoint['norm_d'].append(
                                    get_total_norm(discriminator.parameters()))
                              d_steps_left -= 1
                        elif g_steps_left > 0:
                              step_type = 'g'
                              losses_g = generator_step(args, batch, generator, 
                                                            discriminator, g_loss_fn, 
                                                            optimizer_g)
                              checkpoint['norm_g'].append(
                                    get_total_norm(generator.parameters())
                              )
                              g_steps_left -= 1

                        # Skip the rest if we are not at the end of an iteration
                        if d_steps_left > 0 or g_steps_left > 0:
                              continue
                  
                        # Maybe save loss
                        if t % args.print_every == 0:
                              logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                              for k, v in sorted(losses_d.items()):
                                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                                    checkpoint['D_losses'][k].append(v)
                              for k, v in sorted(losses_g.items()):
                                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                                    checkpoint['G_losses'][k].append(v)
                              checkpoint['losses_ts'].append(t)
                        
                        # Maybe save a checkpoint                  
                        if t > 0 and t % args.checkpoint_every == 0:
                              saveCheckpoint()

                        t += 1
                        d_steps_left = args.d_steps
                        g_steps_left = args.g_steps
                        if t >= args.num_iterations or epoch % args.num_epochs == 0:
                              saveCheckpoint()
                              logger.info("Training process is done.")
                              break

                  if epoch % args.num_epochs == 0:
                        saveCheckpoint()
                        logger.info("Training process is done.")
                        break


def discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d
):
      batch = [tensor.cuda() for tensor in batch]
      (frame_seq, typeID_seq, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
      non_linear_ped, loss_mask, seq_start_end) = batch

      losses = {}
      loss = torch.zeros(1).to(pred_traj_gt)
      generator_out = generator(obs_traj, obs_traj_rel, typeID_seq, seq_start_end)

      pred_traj_fake_rel = generator_out
      pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

      traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
      traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
      traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
      traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    
      scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
      scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

      # Compute loss with optional gradient penalty
      data_loss = d_loss_fn(scores_real, scores_fake)
      losses['D_data_loss'] = data_loss.item()
      loss += data_loss
      losses['D_total_loss'] = loss.item()

      optimizer_d.zero_grad()
      loss.backward()
      if args.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), 
                                    args.clipping_threshold_d)
      optimizer_d.step()

      return losses

def generator_step(
    args, batch, generator, discriminator, g_loss_fn, optimizer_g
):
      batch = [tensor.cuda() for tensor in batch]
      (frame_seq, typeID_seq, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
      non_linear_ped, loss_mask, seq_start_end) = batch

      losses = {}
      loss = torch.zeros(1).to(pred_traj_gt)
      g_l2_loss_rel = []

      loss_mask = loss_mask[:, args.obs_len:]

      for _ in range(args.best_k):
            generator_out = generator(obs_traj, obs_traj_rel, typeID_seq, seq_start_end)
            pred_traj_fake_rel = generator_out
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            if args.l2_loss_weight > 0:
                  g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                              pred_traj_fake_rel,
                              pred_traj_gt_rel,
                              loss_mask,
                              mode='raw'
                        )
                  )

      g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
      if args.l2_loss_weight > 0:
            g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
            for start, end in seq_start_end.data:
                  _g_l2_loss_rel = g_l2_loss_rel[start:end]
                  _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
                  _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                  loss_mask[start:end])
                  g_l2_loss_sum_rel += _g_l2_loss_rel
            losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
            loss += g_l2_loss_sum_rel

      traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
      traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

      scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
      discriminator_loss = g_loss_fn(scores_fake)

      loss += discriminator_loss
      losses['G_discriminator_loss'] = discriminator_loss.item()
      losses['G_total_loss'] = loss.item()

      optimizer_g.zero_grad()
      loss.backward()
      if args.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(
                  generator.parameters(), args.clipping_threshold_g
            )
      optimizer_g.step()

      return losses

def check_accuracy(
      args, loader, generator, discriminator, d_loss_fn, limit=False
):
      d_losses = []
      metrics = {}
      g_l2_losses_abs, g_l2_losses_rel = ([], ) * 2
      disp_error, disp_error_l, disp_error_nl = ([], ) * 3
      f_disp_error, f_disp_error_l, f_disp_error_nl = ([], ) * 3
      total_traj, total_traj_l, total_traj_nl = 0, 0, 0
      loss_mask_sum = 0
      generator.eval()
      with torch.no_grad():
            for batch in loader:
                  batch = [tensor.cuda() for tensor in batch]
                  (frame_seq, typeID_seq, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                  non_linear_ped, loss_mask, seq_start_end) = batch
                  linear_ped = 1 - non_linear_ped
                  loss_mask = loss_mask[:, args.obs_len:]
                  
                  pred_traj_fake_rel = generator(
                        obs_traj, obs_traj_rel, typeID_seq, seq_start_end
                  )
                  pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                  g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                        pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, 
                        pred_traj_fake_rel, loss_mask
                  )
                  ade, ade_l, ade_nl = cal_ade(
                        pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
                  )

                  fde, fde_l, fde_nl = cal_fde(
                        pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
                  )

                  traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
                  traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
                  traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
                  traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

                  scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
                  scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

                  d_loss = d_loss_fn(scores_real, scores_fake)
                  d_losses.append(d_loss.item())

                  g_l2_losses_abs.append(g_l2_loss_abs.item())
                  g_l2_losses_rel.append(g_l2_loss_rel.item())
                  disp_error.append(ade.item())
                  disp_error_l.append(ade_l.item())
                  disp_error_nl.append(ade_nl.item())
                  f_disp_error.append(fde.item())
                  f_disp_error_l.append(fde_l.item())
                  f_disp_error_nl.append(fde_nl.item())

                  loss_mask_sum += torch.numel(loss_mask.data)
                  total_traj += pred_traj_gt.size(1)
                  total_traj_l += torch.sum(linear_ped).item()
                  total_traj_nl += torch.sum(non_linear_ped).item()
                  if limit and total_traj >= args.num_samples_check:
                        break

      metrics['d_loss'] = sum(d_losses) / len(d_losses)
      metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
      metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

      metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
      metrics['fde'] = sum(f_disp_error) / total_traj
      if total_traj_l != 0:
            metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
            metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
      else:
            metrics['ade_l'] = 0
            metrics['fde_l'] = 0
      if total_traj_nl != 0:
            metrics['ade_nl'] = sum(disp_error_nl) / (
                  total_traj_nl * args.pred_len)
            metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
      else:
            metrics['ade_nl'] = 0
            metrics['fde_nl'] = 0

      generator.train()
      return metrics

def dir_check_accuracy(
      args, loader, generator, discriminator, d_loss_fn, limit=False
):
      d_losses = []
      metrics = {}
      g_direction_losses = []
      disp_error, disp_error_l, disp_error_nl = ([], ) * 3
      f_disp_error, f_disp_error_l, f_disp_error_nl = ([], ) * 3
      total_traj, total_traj_l, total_traj_nl = 0, 0, 0
      loss_mask_sum = 0
      generator.eval()
      with torch.no_grad():
            for batch in loader:
                  batch = [tensor.cuda() for tensor in batch]
                  (frame_seq, typeID_seq, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, 
                  non_linear_ped, loss_mask, seq_start_end) = batch
                  linear_ped = 1 - non_linear_ped
                  loss_mask = loss_mask[:, args.obs_len:]
                  
                  obs_dir = obs_to_direction(obs_traj)
                  pred_dir = pred_to_direction(obs_traj, pred_traj_gt)
                  pred_dir_fake = generator(
                        obs_dir, typeID_seq, seq_start_end
                  )

                  g_direction_loss = cal_direction_losses(pred_dir, pred_dir_fake, loss_mask)
                  ade, ade_l, ade_nl = cal_direction_ade(
                        pred_dir, pred_dir_fake, linear_ped, non_linear_ped
                  )
                  
                  fde, fde_l, fde_nl = cal_direction_fde(
                        pred_dir, pred_dir_fake, linear_ped, non_linear_ped
                  )

                  dir_real = torch.cat([obs_dir, pred_dir], dim=0)
                  dir_fake = torch.cat([obs_dir, pred_dir_fake], dim=0)

                  scores_fake = discriminator(dir_fake, typeID_seq, seq_start_end)
                  scores_real = discriminator(dir_real, typeID_seq, seq_start_end)

                  d_loss = d_loss_fn(scores_real, scores_fake)
                  d_losses.append(d_loss.item())

                  g_direction_losses.append(g_direction_loss.item())
                  disp_error.append(ade.item())
                  disp_error_l.append(ade_l.item())
                  disp_error_nl.append(ade_nl.item())
                  f_disp_error.append(fde.item())
                  f_disp_error_l.append(fde_l.item())
                  f_disp_error_nl.append(fde_nl.item())

                  loss_mask_sum += torch.numel(loss_mask.data)
                  total_traj += pred_traj_gt.size(1)
                  total_traj_l += torch.sum(linear_ped).item()
                  total_traj_nl += torch.sum(non_linear_ped).item()
                  if limit and total_traj >= args.num_samples_check:
                        break

      metrics['d_loss'] = sum(d_losses) / len(d_losses)
      metrics['g_direction_loss'] = sum(g_direction_losses) / loss_mask_sum

      metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
      metrics['fde'] = sum(f_disp_error) / total_traj

      if total_traj_l != 0:
            metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
            metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
      else:
            metrics['ade_l'] = 0
            metrics['fde_l'] = 0
      if total_traj_nl != 0:
            metrics['ade_nl'] = sum(disp_error_nl) / (
                  total_traj_nl * args.pred_len)
            metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
      else:
            metrics['ade_nl'] = 0
            metrics['fde_nl'] = 0

      generator.train()
      return metrics

def cal_direction_losses(pred_dir_gt, pred_dir_fake, loss_mask):
      g_direction_loss = direction_loss(
            pred_dir_fake, pred_dir_gt, loss_mask, mode='sum'
      )
      return g_direction_loss

def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, 
    loss_mask
):
      g_l2_loss_abs = l2_loss(
            pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
      )
      g_l2_loss_rel = l2_loss(
            pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
      )
      return g_l2_loss_abs, g_l2_loss_rel

def cal_direction_ade(pred_dir_gt, pred_dir_fake, linear_ped, non_linear_ped):
      ade = directional_error(pred_dir_fake, pred_dir_gt)
      ade_l = directional_error(pred_dir_fake, pred_dir_gt, linear_ped)
      ade_nl = directional_error(pred_dir_fake, pred_dir_gt, non_linear_ped)
      return ade, ade_l, ade_nl

def cal_direction_fde(pred_dir_gt, pred_dir_fake, linear_ped, non_linear_ped):
      fde = final_directional_error(pred_dir_fake, pred_dir_gt)
      fde_l = final_directional_error(pred_dir_fake, pred_dir_gt, linear_ped)
      fde_nl = final_directional_error(pred_dir_fake, pred_dir_gt, non_linear_ped)
      return fde, fde_l, fde_nl

def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
      ade = displacement_error(pred_traj_fake, pred_traj_gt)
      ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
      ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
      return ade, ade_l, ade_nl

def cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
      fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
      fde_l = final_displacement_error(
            pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
      )
      fde_nl = final_displacement_error(
            pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
      )
      return fde, fde_l, fde_nl

if __name__ == '__main__':
      args = parser.parse_args()
      main(args)

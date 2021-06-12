import os
import math
import torch.nn as nn
import torch
import torch.optim as optim
from attrdict import AttrDict

from sgan.losses import direction_loss

def obs_to_direction(obs_traj):
      obs_len = obs_traj.shape[0]
      batch = obs_traj.shape[1]
      obs_direction = torch.zeros(obs_len, batch, 13)
      
      obs_direction[0, :, 12] = 1
      
      for i in range(1, obs_len):
            for j in range(batch):
                  x, y = obs_traj[i, j, 0] - obs_traj[i-1, j, 0], obs_traj[i, j, 1] - obs_traj[i-1, j, 1]
                  if x==0 and y==0:
                        obs_direction[i, j, 12] = 1
                  else:
                        atan2 = math.atan2(y, x)
                        k = int((1+atan2/math.pi)*6)%12
                        obs_direction[i, j, k] = 1

      return obs_direction.cuda()

def pred_to_direction(obs_traj, pred_traj):
      pred_len = pred_traj.shape[0]
      batch = pred_traj.shape[1]
      pred_direction = torch.zeros(pred_len, batch, 13)

      for i in range(batch):
            x, y = pred_traj[0, i, 0] - obs_traj[-1, i, 0], pred_traj[0, i, 1] - obs_traj[-1, i, 1]
            if x==0 and y==0:
                  pred_direction[0, i, 12] = 1
            else:
                  atan2 = math.atan2(y, x)
                  k = int((1+atan2/math.pi)*6)%12
                  pred_direction[0, i, k] = 1
            
      for i in range(1, pred_len):
            for j in range(batch):
                  x, y = pred_traj[i, j, 0] - pred_traj[i-1, j, 0], pred_traj[i, j, 1] - pred_traj[i-1, j, 1]
                  if x==0 and y==0:
                        pred_direction[i, j, 12] = 1
                  else:
                        atan2 = math.atan2(y, x)
                        k = int((1+atan2/math.pi)*6)%12
                        pred_direction[i, j, k] = 1
                              
      return pred_direction.cuda()

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
      layers = []
      for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            if batch_norm:
                  layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                  layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                  layers.append(nn.LeakyReLU())
            if dropout > 0:
                  layers.append(nn.Dropout(p=dropout))
      return nn.Sequential(*layers)

def get_noise(shape, noise_type):
      if noise_type == 'gaussian':
            return torch.randn(*shape).cuda()
      elif noise_type == 'uniform':
            return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
      raise ValueError('Unrecognized noise type "%s"' % noise_type)

def direction_discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d
):
      batch = [tensor.cuda() for tensor in batch]
      (frame_seq, typeID_seq, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
      non_linear_ped, loss_mask, seq_start_end) = batch

      losses = {}
      loss = torch.zeros(1).to(pred_traj_gt)

      obs_dir = obs_to_direction(obs_traj)
      pred_dir = pred_to_direction(obs_traj, pred_traj_gt)
      
      pred_dir_fake = generator(obs_dir, typeID_seq, seq_start_end)
      dir_real = torch.cat([obs_dir, pred_dir], dim=0)
      dir_fake = torch.cat([obs_dir, pred_dir_fake], dim=0)

      scores_fake = discriminator(dir_fake, typeID_seq, seq_start_end)
      scores_real = discriminator(dir_real, typeID_seq, seq_start_end)

      # Compute loss with optional gradient penalty
      data_loss = d_loss_fn(scores_real, scores_fake)
      losses['D_data_loss'] = data_loss.item()
      loss += data_loss
      losses['D_total_loss'] = loss.item()

      optimizer_d.zero_grad()
      loss.backward()
      if args.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(),
                  args.clipping_threshold_d
            )
      optimizer_d.step()

      return losses

def direction_generator_step(
    args, batch, generator, discriminator, g_loss_fn, optimizer_g
):
      batch = [tensor.cuda() for tensor in batch]
      (frame_seq, typeID_seq, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, 
      non_linear_ped, loss_mask, seq_start_end) = batch
      losses = {}
      loss = torch.zeros(1).to(pred_traj_gt)
      g_direction_loss_rel = []

      loss_mask = loss_mask[:, args.obs_len:]

      obs_dir = obs_to_direction(obs_traj)
      pred_dir = pred_to_direction(obs_traj, pred_traj_gt)

      for _ in range(args.best_k):
            generator_out = generator(obs_dir, typeID_seq, seq_start_end)

            pred_dir_fake = generator_out

            if args.direction_loss_weight > 0:
                  g_direction_loss_rel.append(args.direction_loss_weight * direction_loss(
                  pred_dir_fake,
                  pred_dir,
                  loss_mask,
                  mode='raw'))

      g_direction_loss_sum_rel = torch.zeros(1).to(pred_dir)
      if args.direction_loss_weight > 0:
            g_direction_loss_rel = torch.stack(g_direction_loss_rel, dim=1)
            for start, end in seq_start_end.data:
                  _g_direction_loss_rel = g_direction_loss_rel[start:end]
                  _g_direction_loss_rel = torch.sum(_g_direction_loss_rel, dim=0)
                  _g_direction_loss_rel = torch.min(_g_direction_loss_rel) / torch.sum(
                  loss_mask[start:end])
                  g_direction_loss_sum_rel += _g_direction_loss_rel
            losses['G_direction_loss_rel'] = g_direction_loss_sum_rel.item()
            loss += g_direction_loss_sum_rel

      dir_fake = torch.cat([obs_dir, pred_dir_fake], dim=0)

      scores_fake = discriminator(dir_fake, typeID_seq, seq_start_end)
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

class Encoder(nn.Module):
      def __init__(
            self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
            dropout=0.0,
            obs_len=8, pred_len=8
      ):
            super(Encoder, self).__init__()

            self.mlp_dim = 1024
            self.h_dim = h_dim
            self.embedding_dim = embedding_dim
            self.num_layers = num_layers
            self.obs_len = obs_len
            self.pred_len = pred_len

            self.encoder = nn.LSTM(
                  embedding_dim, h_dim, num_layers, dropout=dropout
            )

            self.spatial_embedding = nn.Linear(14, embedding_dim)

      def init_hidden(self, batch):
            return (
                  torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
                  torch.zeros(self.num_layers, batch, self.h_dim).cuda()
            )

      def forward(self, obs_dir, typeID_seq, encoder_type="generator"):
            """
            Inputs:
            - obs_dir: Tensor of shape (obs_len, batch, 2)
            - typeID_seq: Tensor of shape (obs_len + pred_len, batch, 2)
            Output:
            - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
            """
            # Encode observed Trajectory
            batch = obs_dir.size(1)
            if encoder_type=="generator":
                  obs_type_ID_seq = torch.split(typeID_seq, [self.obs_len, self.pred_len], dim=0)[0].cuda()
            else:
                  obs_type_ID_seq = typeID_seq.cuda()
            input_info = torch.cat((obs_dir, obs_type_ID_seq), 2)
            input_info_embedding = self.spatial_embedding(input_info.view(-1, 14))
            input_info_embedding = input_info_embedding.view(
                  -1, batch, self.embedding_dim
            )
            state_tuple = self.init_hidden(batch)
            output, state = self.encoder(input_info_embedding, state_tuple)
            final_h = state[0]
            return final_h

class Decoder(nn.Module):
      """Decoder is part of TrajectoryGenerator"""
      def __init__(
            self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
            dropout=0.0, bottleneck_dim=1024,
            activation='relu', batch_norm=True
      ):
            super(Decoder, self).__init__()

            self.seq_len = seq_len
            self.mlp_dim = mlp_dim
            self.h_dim = h_dim
            self.embedding_dim = embedding_dim
            
            self.decoder = nn.LSTM(
                  embedding_dim, h_dim, num_layers, dropout=dropout
            )
            
            self.spatial_embedding = nn.Linear(13, embedding_dim)
            self.hidden2dir = nn.Linear(h_dim, 13)
            self.activation = nn.Sigmoid()

      def forward(self, last_dir, state_tuple, seq_start_end):
            """
            Inputs:
            - last_dir: Tensor of shape (batch, 2)
            - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
            - seq_start_end: A list of tuples which delimit sequences within batch
            Output:
            - pred_dir: tensor of shape (self.seq_len, batch, 2)
            """
            batch = last_dir.size(0)
            pred_dir_fake_rel = []
            decoder_input = self.spatial_embedding(last_dir)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)

            for _ in range(self.seq_len):
                  output, state_tuple = self.decoder(decoder_input, state_tuple)
                  rel_dir = self.hidden2dir(output.view(-1, self.h_dim))
                  
                  embedding_input = rel_dir

                  decoder_input = self.spatial_embedding(embedding_input)
                  decoder_input = decoder_input.view(1, batch, self.embedding_dim)
                  pred_dir_fake_rel.append(rel_dir.view(batch, -1))

            pred_dir_fake_rel = torch.stack(pred_dir_fake_rel, dim=0)
            pred_dir_fake_rel = self.activation(pred_dir_fake_rel)
            return pred_dir_fake_rel, state_tuple[0]

class DirectionGenerator(nn.Module):
      def __init__(
            self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
            decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
            noise_type='gaussian', noise_mix_type='ped',
            dropout=0.0, bottleneck_dim=1024,
            activation='relu', batch_norm=True,
      ):
            super(DirectionGenerator, self).__init__()

            self.obs_len = obs_len
            self.pred_len = pred_len
            self.mlp_dim = mlp_dim
            self.encoder_h_dim = encoder_h_dim
            self.decoder_h_dim = decoder_h_dim
            self.embedding_dim = embedding_dim
            self.noise_dim = noise_dim
            self.num_layers = num_layers
            self.noise_type = noise_type
            self.noise_mix_type = noise_mix_type
            self.noise_first_dim = 0
            self.bottleneck_dim = 1024

            self.encoder = Encoder(
                  embedding_dim=embedding_dim,
                  h_dim=encoder_h_dim,
                  mlp_dim=mlp_dim,
                  num_layers=num_layers,
                  dropout=dropout,
                  obs_len=obs_len,
                  pred_len=pred_len
            )

            self.decoder = Decoder(
                  pred_len,
                  embedding_dim=embedding_dim,
                  h_dim=decoder_h_dim,
                  mlp_dim=mlp_dim,
                  num_layers=num_layers,
                  dropout=dropout,
                  bottleneck_dim=bottleneck_dim,
                  activation=activation,
                  batch_norm=batch_norm,
            )

            if self.noise_dim[0] == 0:
                  self.noise_dim = None
            else:
                  self.noise_first_dim = noise_dim[0]

            # Decoder Hidden
            input_dim = encoder_h_dim

            if self.mlp_decoder_needed():
                  mlp_decoder_context_dims = [
                        input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
                  ]

                  self.mlp_decoder_context = make_mlp(
                        mlp_decoder_context_dims,
                        activation=activation,
                        batch_norm=batch_norm,
                        dropout=dropout
                  )

      def add_noise(self, _input, seq_start_end, user_noise=None):
            """
            Inputs:
            - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
            - seq_start_end: A list of tuples which delimit sequences within batch.
            - user_noise: Generally used for inference when you want to see
            relation between different types of noise and outputs.
            Outputs:
            - decoder_h: Tensor of shape (_, decoder_h_dim)
            """
            if not self.noise_dim:
                  return _input

            if self.noise_mix_type == 'global':
                  noise_shape = (seq_start_end.size(0), ) + self.noise_dim
            else:
                  noise_shape = (_input.size(0), ) + self.noise_dim

            if user_noise is not None:
                  z_decoder = user_noise
            else:
                  z_decoder = get_noise(noise_shape, self.noise_type)

            if self.noise_mix_type == 'global':
                  _list = []
                  for idx, (start, end) in enumerate(seq_start_end):
                        start = start.item()
                        end = end.item()
                        _vec = z_decoder[idx].view(1, -1)
                        _to_cat = _vec.repeat(end - start, 1)
                        _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
                  decoder_h = torch.cat(_list, dim=0)
                  return decoder_h

            decoder_h = torch.cat([_input, z_decoder], dim=1)

            return decoder_h

      def mlp_decoder_needed(self):
            if (
                  self.noise_dim or self.pooling_type or
                  self.encoder_h_dim != self.decoder_h_dim
            ):
                  return True
            else:
                  return False

      def forward(self, obs_dir, typeID_seq, seq_start_end, user_noise=None):
            """
            Inputs:
            - obs_dir: Tensor of shape (obs_len, batch, 13)
            - typeID_seq: Tensor of shape (obs_len + pred_len, batch, 2)
            - seq_start_end: A list of tuples which delimit sequences within batch.
            - user_noise: Generally used for inference when you want to see
            relation between different types of noise and outputs.
            Output:
            - pred_dir_rel: Tensor of shape (self.pred_len, batch, 13)
            """
            batch = obs_dir.size(1)
            # Encode seq
            final_encoder_h = self.encoder(obs_dir, typeID_seq, "generator")
            # Pool States
            mlp_decoder_context_input = final_encoder_h.view(
                  -1, self.encoder_h_dim)

            # Add Noise
            if self.mlp_decoder_needed():
                  noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
            else:
                  noise_input = mlp_decoder_context_input
            decoder_h = self.add_noise(
                  noise_input, seq_start_end, user_noise=user_noise)
            decoder_h = torch.unsqueeze(decoder_h, 0)

            decoder_c = torch.zeros(
                  self.num_layers, batch, self.decoder_h_dim
            ).cuda()

            state_tuple = (decoder_h, decoder_c)
            last_dir = obs_dir[-1]
            # Predict Trajectory

            decoder_out = self.decoder(
                  last_dir,
                  state_tuple,
                  seq_start_end,
            )
            pred_dir_fake, final_decoder_h = decoder_out
                       
            return pred_dir_fake

class DirectionDiscriminator(nn.Module):
      def __init__(
            self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
            num_layers=1, activation='relu', batch_norm=True, dropout=0.0            
      ):
            super(DirectionDiscriminator, self).__init__()

            self.obs_len = obs_len
            self.pred_len = pred_len
            self.seq_len = obs_len + pred_len
            self.mlp_dim = mlp_dim
            self.h_dim = h_dim

            self.encoder = Encoder(
                  embedding_dim=embedding_dim,
                  h_dim=h_dim,
                  mlp_dim=mlp_dim,
                  num_layers=num_layers,
                  dropout=dropout,
                  obs_len=obs_len,
                  pred_len=pred_len
            )

            real_classifier_dims = [h_dim, mlp_dim, 1]
            self.real_classifier = make_mlp(
                  real_classifier_dims,
                  activation=activation,
                  batch_norm=batch_norm,
                  dropout=dropout
            )

      def forward(self, direction, typeID_seq, seq_start_end=None):
            """
            Inputs:
            - direction: Tensor of shape (obs_len + pred_len, batch, 13)
            - typeID_seq: Tensor of shape (obs_len + pred_len, batch, 2)
            - seq_start_end: A list of tuples which delimit sequences within batch
            Output:
            - scores: Tensor of shape (batch,) with real/fake scores
            """
            final_h = self.encoder(direction, typeID_seq, "discriminator")
            # Note: In case of 'global' option we are using start_pos as opposed to
            # end_pos. The intution being that hidden state has the whole
            # trajectory and relative postion at the start when combined with
            # trajectory information should help in discriminative behavior.
            classifier_input = final_h.squeeze()
            scores = self.real_classifier(classifier_input)
            return scores

class DirectionModule():
      def __init__(self, args=None):
            # self.obs_len = args.obs_len
            # self.pred_len = args.pred_len
            self.checkpoint = torch.load(os.path.join(args.dir_output_dir, '%s_with_model.pt' % args.dir_checkpoint_name))
            self.generator = self.get_generator(self.checkpoint)
            for param in self.generator.parameters():
                  param.requires_grad = False

      def get_generator(self, checkpoint=None):
            args = AttrDict(checkpoint['args'])
            generator = DirectionGenerator(
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
                  batch_norm=args.batch_norm)
            generator.load_state_dict(checkpoint['g_state'])
            generator.cuda()
            generator.train()
            return generator
      
      def __call__(self, obs_dir, typeID_seq, seq_start_end):
            return self.forward(obs_dir, typeID_seq, seq_start_end)

      def forward(self, obs_dir, typeID_seq, seq_start_end):
            pred_dir = self.generator(obs_dir, typeID_seq, seq_start_end)
            pred_dir = pred_dir.permute(1, 0, 2)
            pred_dir = torch.flatten(pred_dir, start_dim=1)
            return pred_dir.cuda()

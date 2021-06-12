import torch
import torch.nn as nn
import gc

from sgan.directionmod import DirectionModule, obs_to_direction
from sgan.sceneseg import SceneModule

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

class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0,args=None
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.use_gru = args.use_gru
        
        if self.use_gru:
            self.encoder = nn.GRU(
                embedding_dim, h_dim, num_layers, dropout=dropout
            )
        else:
            self.encoder = nn.LSTM(
                embedding_dim, h_dim, num_layers, dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        if self.use_gru:
            output, state = self.encoder(obs_traj_embedding, state_tuple[0])
            final_h = state
        else:
            output, state = self.encoder(obs_traj_embedding, state_tuple)
            final_h = state[0]
        return final_h

class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True,
        args=None
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.module_num = 0

        self.pool_net_on = args.g_pool_net_on
        self.direction_on = args.g_direction_on
        self.typeID_on = args.g_typeID_on 
        self.scene_on = args.g_scene_on

        self.use_gru = args.use_gru
        if self.use_gru:
            self.decoder = nn.GRU(
                embedding_dim, h_dim, num_layers, dropout=dropout
            )
        else:
            self.decoder = nn.LSTM(
                embedding_dim, h_dim, num_layers, dropout=dropout
            )
        
        if pool_every_timestep:
            if self.pool_net_on:
                self.module_num += 1
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )

            if self.direction_on:
                self.module_num += 1
                self.direction_generator = DirectionModule(args=args)
                self.direction_context = make_mlp(
                    [13, 512, bottleneck_dim],
                    activation=activation, 
                    batch_norm=batch_norm, 
                    dropout=dropout
                )
            
            if self.typeID_on:
                self.module_num += 1
                self.typeID_embedding=nn.Linear(1, self.h_dim)
                self.typeID_context = make_mlp(
                    [self.h_dim, 512, bottleneck_dim],
                    activation=activation, 
                    batch_norm=batch_norm, 
                    dropout=dropout
                )

            if self.scene_on:
                self.module_num += 1
                self.scene_module=SceneModule(
                    seg_rec_size=args.seg_rec_size, 
                    filename=args.seg_scene_file,
                    seg_scale=args.seg_scale,
                    scene_object_types=args.scene_object_types
                )
                self.scene_context = nn.Linear(args.scene_object_types*3, bottleneck_dim)

            mlp_dims = [h_dim + bottleneck_dim * self.module_num, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
            # print("h_dim:", h_dim, "bottleneck_dim:", bottleneck_dim, "mlp_dim:", mlp_dim)

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, obs_traj, obs_traj_rel, typeID_seq, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            if self.use_gru:
                output, state_tuple = self.decoder(decoder_input, state_tuple[0])
            else:
                output, state_tuple = self.decoder(decoder_input, state_tuple)

            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                if self.use_gru:
                    decoder_h = state_tuple
                else:
                    decoder_h = state_tuple[0]
                    
                mlp_context_input = decoder_h.view(-1, self.h_dim)
                
                if self.pool_net_on:
                    pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                    # print("P:", pool_h.shape)
                    mlp_context_input = torch.cat(
                        [mlp_context_input, pool_h], dim=1)

                if self.direction_on:
                    obs_dir = obs_to_direction(obs_traj_rel)
                    direction_context = self.direction_generator(obs_dir, typeID_seq, seq_start_end)
                    direction_em = self.direction_context(direction_context[:,0:13])
                    # print("D:", direction_em.shape)
                    mlp_context_input = torch.cat(
                        [mlp_context_input, direction_em], dim=1)

                if self.typeID_on:
                    typeID_raw = typeID_seq[-1,:,:]
                    typeID_em = self.typeID_embedding(typeID_raw)
                    typeID_h = self.typeID_context(typeID_em)
                    # print("T:", typeID_h.shape)
                    mlp_context_input = torch.cat(
                        [mlp_context_input, typeID_h], dim=1)
                
                if self.scene_on:
                    scene_info = self.scene_module(torch.unsqueeze(last_pos,0))
                    scene_h = self.scene_context(scene_info)
                    scene_h = torch.squeeze(scene_h, 0)                    
                    # print("S:", scene_h.shape)
                    mlp_context_input = torch.cat(
                        [mlp_context_input, scene_h], dim=1)
                    
                decoder_h = self.mlp(mlp_context_input)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                if self.use_gru:
                    state_tuple = (decoder_h, state_tuple)
                else:
                    state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]

class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h

class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped',
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True,
        args=None
    ):
        super(TrajectoryGenerator, self).__init__()

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
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = bottleneck_dim
        
        self.pool_net_on = args.g_pool_net_on
        self.direction_on = args.g_direction_on
        self.typeID_on = args.g_typeID_on
        self.scene_on = args.g_scene_on
        self.module_num = 0
        self.h_dim = encoder_h_dim

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
            args=args
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            args=args
        )

        if self.pool_net_on:
            self.module_num += 1
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )
            
        if self.direction_on:
            self.module_num += 1
            self.direction_generator = DirectionModule(args=args)
            self.direction_context = make_mlp(
                [self.pred_len*13, 512, bottleneck_dim],
                activation=activation, 
                batch_norm=batch_norm, 
                dropout=dropout
            )
        
        if self.typeID_on:
            self.module_num += 1
            self.typeID_embedding=nn.Linear(1, self.h_dim)
            self.typeID_context = make_mlp(
                [self.h_dim, 512, bottleneck_dim],
                activation=activation, 
                batch_norm=batch_norm, 
                dropout=dropout
            )

        if self.scene_on:
            self.module_num += 1
            self.scene_module=SceneModule(
                seg_rec_size=args.seg_rec_size, 
                filename=args.seg_scene_file,
                seg_scale=args.seg_scale,
                scene_object_types=args.scene_object_types
            )
            self.scene_context = nn.Linear(obs_len * args.scene_object_types * 3, bottleneck_dim)
            
        #* LSTM 取代 class encoder
        self.lstm_encoder = nn.LSTM(self.h_dim, self.h_dim, num_layers, dropout=dropout)
        #* 給 LSTM 用的
        self.feature_hidden_embedding = nn.Linear(self.h_dim * self.module_num, self.h_dim)
        #* 給 obj_rel 用的
        self.traj_rel_embedding = nn.Linear(2, self.h_dim)
        #* 給 lstm state h 用的
        self.lstm_state_h_embedding = nn.Linear(self.h_dim * 2, self.h_dim)  

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        input_dim = encoder_h_dim + bottleneck_dim * self.module_num
        # print("encoder_h_dim:",encoder_h_dim," bottleneck_dim:",bottleneck_dim,", input_dim:", input_dim)
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
            self.noise_dim or 
            self.direction_on or self.typeID_on or self.pool_net_on or self.scene_on or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, typeID_seq, seq_start_end, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        
        # Pool States
        if self.module_num == 0:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)
            end_pos = obs_traj[-1, :, :]
            # print("mlp_decoder_context_input:", mlp_decoder_context_input.shape)

            if self.pool_net_on:
                pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
                # print("pool_h:", pool_h.shape)
                mlp_decoder_context_input = torch.cat(
                    [mlp_decoder_context_input, pool_h], dim=1)

            if self.direction_on:
                obs_dir = obs_to_direction(obs_traj_rel)
                direction_context = self.direction_generator(obs_dir, typeID_seq, seq_start_end)
                direction_em = self.direction_context(direction_context)
                # print("direction_em:", direction_em.shape)
                mlp_decoder_context_input = torch.cat(
                    [mlp_decoder_context_input, direction_em], dim=1)

            if self.typeID_on:
                typeID_raw = typeID_seq[-1,:,:]
                typeID_em = self.typeID_embedding(typeID_raw)
                typeID_h = self.typeID_context(typeID_em)
                # print("typeID_h:", typeID_h.shape)
                mlp_decoder_context_input = torch.cat(
                    [mlp_decoder_context_input, typeID_h], dim=1)
            
            if self.scene_on:
                scene_info = self.scene_module(obs_traj)
                scene_info = torch.flatten(scene_info.permute(1, 0, 2), start_dim=1)
                scene_h = self.scene_context(scene_info)
                # print("scene_h:", scene_h.shape)
                mlp_decoder_context_input = torch.cat(
                    [mlp_decoder_context_input, scene_h], dim=1)

        # print("mlp_decoder_context_input", mlp_decoder_context_input.shape)

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
        # Predict Trajectory

        decoder_out = self.decoder(
            obs_traj,
            obs_traj_rel,
            typeID_seq,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        gc.collect()
        
        return pred_traj_fake_rel.cuda()

class TrajectoryDiscriminator(nn.Module): 
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local', args=None
    ):
        super(TrajectoryDiscriminator, self).__init__()
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.use_gru = args.use_gru

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
            args = args
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )
        
    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        
        final_h = self.encoder(traj_rel)
        
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.

        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )

        scores = self.real_classifier(classifier_input)
        gc.collect()
        return scores

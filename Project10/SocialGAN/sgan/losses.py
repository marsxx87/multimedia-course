import torch
import random
import math

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)

def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake

def direction_loss(pred_dir, pred_dir_gt, loss_mask, mode='average'):
    """
    Input:
    - pred_dir: Tensor of shape (seq_len, batch, 13). Predicted direction.
    - pred_dir_gt: Tensor of shape (seq_len, batch, 13). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: directional loss depending on mode
    """
    seq_len, batch, _ = pred_dir.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_dir_gt.permute(1, 0, 2) - pred_dir.permute(1, 0, 2))**2)

    # Turning size of pred_dir and pred_dir_gt into (batch, seq_len, 13)
    # pred_dir, pred_dir_gt = pred_dir.permute(1, 0, 2), pred_dir_gt.permute(1, 0, 2)
    # error = torch.zeros_like(pred_dir)
    # exp_pred_dir = torch.exp(pred_dir)
    # for i in range(batch):
    #     for j in range(seq_len):
    #         sum_exp_pred = torch.sum(exp_pred_dir[i, j])
    #         x_class = torch.dot(pred_dir[i, j], pred_dir_gt[i, j])
    #         # if x_class == 0:
    #         #     print("GT:", pred_dir_gt[i, j])
    #         #     print("Pred:", pred_dir[i, j])
    #         #     print("x_class:", x_class)
    #         #     print("Sig(exp(pred)):", sum_exp_pred)
    #         #     input("Error in Loss.")
    #         loss = torch.log(sum_exp_pred) - x_class
    #         error[i, j, 0] = error[i, j, 1] = loss

    # loss = (loss_mask.unsqueeze(dim=2) * error)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)
    

def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def directional_error(pred_dir, pred_dir_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_dir: Tensor of shape (seq_len, batch, 13). Predicted direction.
    - pred_dir_gt: Tensor of shape (seq_len, batch, 13). Ground truth direction.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the directional error
    """
    seq_len, batch, _ = pred_dir.size()
    loss = torch.zeros(batch)
    for i in range(seq_len):
        for j in range(batch):
            if pred_dir_gt[i, j, 12]==1:
                continue
            
            angle_gt = 0
            angle = 0            
                                
            for k in range(1, 12):
                if pred_dir_gt[i, j, k] >= 1:
                    angle_gt = k
                if pred_dir[i, j, k] > pred_dir[i, j, angle]:
                    angle = k
            if abs(angle_gt - angle) > 6:
                loss[j] = abs(angle - angle_gt)-6
            else:
                loss[j] = abs(angle - angle_gt)
    
    if consider_ped is not None:
        loss = torch.matmul(loss.cuda(), consider_ped)
        
    if mode == 'sum': #回傳bat 的loss總和
        return torch.sum(loss)
    elif mode == 'raw': #回傳完整loss的陣列
        return loss

def final_directional_error(pred_dir, pred_dir_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_dir: Tensor of shape (seq_len, batch, 13). Predicted direction.
    - pred_dir_gt: Tensor of shape (seq_len, batch, 13). Ground truth direction.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the final directional error
    """
    _, batch_size, _ = pred_dir.size()
    
    loss = torch.zeros(batch_size)
    
    final_pred_dir = pred_dir[-1] - pred_dir[0]
    final_pred_dir_gt = pred_dir_gt[-1] - pred_dir_gt[0]

    for i in range(batch_size):
        if not pred_dir_gt[-1, i, 12]==1:

            angle_gt = 0
            angle = 0
            
            for j in range(12):
                if pred_dir_gt[-1, i, j] >= 1:
                    angle_gt = j
                if pred_dir[-1, i, j] > pred_dir[-1, i, j]:
                    angle = j
                loss[i] = (angle-angle_gt)%6

    if consider_ped is not None:
        loss = torch.matmul(loss.cuda(), consider_ped)
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)

def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)

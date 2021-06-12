import logging
import os
import math
import numpy as np
import pandas as pd
import torch
import math
import time

from torch.utils.data import Dataset
#from PIL import Image

logger = logging.getLogger(__name__)

# 取得getitem資料後，搭配自訂規則，輸出最後結果(訓練資料格式)
def seq_collate(data):
    (frame_seq_list, typeID_seq_list, obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
    non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len  ex:[batch, input_size, seq_len](64筆的總和,2,8) => [seq_len, batch, input_size]
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    #* 加入 frame,typeID,block_rel
    frame_seq = torch.cat(frame_seq_list, dim=0).permute(2, 0, 1)
    typeID_seq = torch.cat(typeID_seq_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    out = [
        frame_seq, typeID_seq, obs_traj, pred_traj, obs_traj_rel,
        pred_traj_rel, non_linear_ped, loss_mask, seq_start_end
    ]

    return tuple(out)

def read_file_csv(_path,
        #資策會資料集用
        #type_list=['Adult', 'AdultUnderUmbrella', 'BicycleWithRider', 'MotorcycleWithRiderWithHelmet',  'MotorcycleWithRiderWithoutHelmet', 'Sedan', 'Taxi', 'PoliceCar', 'SmallTruck', 'ConstructionVehicle', 'Bus', 'SpecialPurposeVehicle']
        #Stanford dataset用
        type_list=['Pedestrian', 'Biker', 'Car', 'Bus', 'Skater', 'Cart']
    ):
    data = []

    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(',')

            if line[3] == type_list[0] or line[3] == type_list[1]:
                traffic_type = 1
            elif line[3] == type_list[2]:
                traffic_type = 0.25
            elif line[3] == type_list[3] or line[3] == type_list[4]:
                traffic_type = 0.5
            elif line[3] == type_list[5] or line[3] == type_list[6] or line[3] == type_list[7]:
                traffic_type = 0.75
            else:
                traffic_type = 1
            
            line = [float(line[2]), float(line[1]), float(line[8]), float(line[9]), traffic_type]
            data.append(line)
    return np.asarray(data)

#* 讀取檔案並切割資料，以及轉換 TypeId
def read_file(_path, delim='\t',
        #資策會資料集用
        #Stanford dataset用
        type_list=['Pedestrian', 'Biker', 'Car', 'Bus', 'Skater', 'Cart']
    ):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '

    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            
            """if line[-1] == type_list[0] or line[-1] == type_list[1]:
                line[-1] = 0
            elif line[-1] == type_list[2]:
                line[-1] = 0.25
            elif line[-1] == type_list[3] or line[-1] == type_list[4]:
                line[-1] = 0.5
            elif line[-1] == type_list[5] or line[-1] == type_list[6] or line[-1] == type_list[7]:
                line[-1] = 0.75
            else:
                line[-1] = 1"""

            line[-1] = 1 #tmp for testing            
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)    
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

# 取得訓練或驗證資料
class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='space'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y> <typeID>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """

        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        
        #* 紀錄 frame, typeID, 紀錄點跟點之間的方向性
        seq_list_frame = []
        seq_list_typeID = []
        
        for path in all_files:
            if "desktop.ini" in path:
                continue
            if path.endswith(r".csv"):
                data = read_file_csv(path)
            else:
                data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / self.skip))
                
            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                #* curr_seq_data = [fram, id, x, y, typeid]
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                #* 去除重複 id，升冪排序
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                #* 紀錄 frame
                curr_seq_frame = np.zeros((len(peds_in_curr_seq),1,self.seq_len))
                #* 紀錄 typeID
                curr_seq_typeId = np.zeros((len(peds_in_curr_seq),1,self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    pred_len, xy_len = curr_ped_seq.shape
                    #* 補上面的寫法有漏洞,假設第一個跟最後一個相減剛好等於長度16,但如果中間有漏追的就會有問題
                    if pred_len != self.seq_len:
                        continue
                    #* 取出 x y 並且轉置 ( 以列的方式存取 [[x0,x1....x15][y0,y1.....y15]] )
                    curr_ped_coordinate = np.transpose(curr_ped_seq[:, 2:-1])
                    
                    #* curr_ped_coordinate = [2,16]
                    #* 取得各步數之間的差（[x1-x0],[x2-x1],[x3-x2]...）
                    rel_curr_ped_seq = np.zeros(curr_ped_coordinate.shape)
                    rel_curr_ped_seq[:, 0] = curr_ped_coordinate[:, 0]
                    rel_curr_ped_seq[:, 1:] = curr_ped_coordinate[:, 1:] - curr_ped_coordinate[:, :-1]
                    _idx = num_peds_considered
                    curr_seq_frame[_idx, :, pad_front:pad_end] = np.transpose(curr_ped_seq[:,0])
                    curr_seq_typeId[_idx,:,pad_front:pad_end] = curr_ped_seq[:,-1]
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_coordinate                                            
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    #* Linear vs Non-Linear Trajectory
                    #* poly_fit 回傳當前資料為線性或非線性（ 0=線性，1=非線性 ）
                    _non_linear_ped.append(poly_fit(curr_ped_coordinate, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                #* 滿足 obs_traj + pred_traj 的步數條件項目，至少要有兩項（含）
                if num_peds_considered >= min_ped:
                    non_linear_ped += _non_linear_ped
                    # 將 frame[0] ~ frame[15] 當中都有出現的 ID 數量存起來，下一筆數量範圍為 frame[1] ~ frame[16]
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    #* 加入 frame, typeID
                    seq_list_frame.append(curr_seq_frame[:num_peds_considered])
                    seq_list_typeID.append(curr_seq_typeId[:num_peds_considered])
                  
        self.num_seq = len(seq_list)
    
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        seq_list_frame = np.concatenate(seq_list_frame,axis=0)
        seq_list_typeID = np.concatenate(seq_list_typeID,axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        #* 加入 frame, typeID
        self.seq_frame = torch.from_numpy(seq_list_frame).type(torch.int)
        self.seq_typeID = torch.from_numpy(seq_list_typeID).type(torch.float)

        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    # 符合長度為16(obs_len + pred_len)的總數
    def __len__(self):
        return self.num_seq

    # 取得指定索引的資料(當該迭代器被呼叫時，會傳出一個同一個範圍（frame[0]~[15] or [1]~[16]）且滿足 16 步的所有 ID 資料)
    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.seq_frame[start:end,:], self.seq_typeID[start:end, :],
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
            ]
        
        return out

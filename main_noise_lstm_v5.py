import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter
from dataset_import import MIMIC_II_Rpeaks_align, read_dataset2memory_align
from dataset_import import MIMIC_II_Rpeaks_align_v2, MIMIC_II_Rpeaks_align_noise_v1
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from optim import ScheduledOptim


from mlp_mixer_attention_v5 import MLPMixer
from metrics import MetricRegression

test_batch_size = 100
batch_size = 256


def read_npz_data(path_npz_data):
    data = np.load(path_npz_data)
    ecg_list = data["ECG"]
    ppg_list = data["PPG"]
    BP_list = data["BP"]
    return ecg_list, ppg_list, BP_list


def train_epoch(train_loader, device, model, optimizer, total_num):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, mininterval=0.5, desc='[### training: ] ', leave=False):
        ecg_ppg, BP = map(lambda x: x.to(device), batch)
        # sys_BP = sys_BP.double()
        # forward
        optimizer.zero_grad()
        pred_BP = model(ecg_ppg)
        pred_BP = pred_BP.reshape(BP.shape)
        # pred_dia_BP = pred_dia_BP.reshape(sys_BP.shape)

        # backward
        # pred_BP = torch.cat((pred_sys_BP, pred_dia_BP), dim=0)
        # real_BP = torch.cat((sys_BP, dia_BP), dim=0)
        loss = F.mse_loss(pred_BP, BP)
        # loss = F.smooth_l1_loss(pred_sys_BP, sys_BP)
        # loss = nn.Huber
        # assert not torch.isnan(loss)
        loss.backward()

        # update
        optimizer.step_and_update_lr()
        total_loss += loss.item()

        # tensorboard
        # summaryWriter.add_scalars("loss", {"train_loss_avg": train_loss_avg, "test_loss_avg": test_loss_avg}, epoch)
        # total_step = i_epoch * (train_num_samples / batch_size) + current_step
        # summaryWriter.add_scalar("loss", {"loss_step": loss}, total_step)
        # current_step += 1

    loss_epoch = total_loss / (total_num / batch_size)
    # summaryWriter.add_scalar("loss", loss_epoch, i_epoch)
    return loss_epoch


def eval_epoch(valid_loader, device, model, total_num):
    all_pred_sys_BP = []
    all_sys_BP = []
    all_BP = []
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=0.5, desc='[### validation: ] ', leave=False):
            ecg_ppg, BP = map(lambda x: x.to(device), batch)

            pred_BP = model(ecg_ppg)
            pred_BP = pred_BP.reshape(BP.shape)
            val_loss = F.mse_loss(pred_BP, BP)

            pred_BP = pred_BP.squeeze().cpu().numpy()
            BP = BP.cpu().numpy()
            all_pred_sys_BP.extend(pred_BP)
            all_BP.extend(BP)
            total_loss += val_loss.item()
    loss_epoch = total_loss / (total_num / test_batch_size)

    return all_pred_sys_BP, all_BP, loss_epoch


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:2')
    else:
        device = torch.device('cpu')
    print("device: ", device)

    train_num_samples = 3600
    signal_length = 128
    start_pos = 200
    model_save_epoch = 20
    dim = 256   # D
    depth = 4    # DP
    token_dim = 256  # TD
    channel_dim = 512  # CD
    dropout = 0.3
    num_classes = 2
    num_peaks = 4
    num_patches = 128
    in_channel = int(num_peaks * 12)

    model_settings = "MLPLSTM_ECGPPG128_noise_NP%s_D%s_DP%s_TD%s_CD%s_dr%s_v50" % (num_peaks, dim,
                                                                            depth, token_dim,
                                                                            channel_dim, int(dropout*10))

    print("model_settings :", model_settings)
    log_dir_base = "./logs/"
    if not os.path.exists(log_dir_base):
        os.mkdir(log_dir_base)
        print("Create log dir: %s" %(log_dir_base))

    log_dir = os.path.join(log_dir_base, model_settings)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        print("Create log dir: %s" %(log_dir))

    model_dir_base = "./models/"
    if not os.path.exists(model_dir_base):
        os.mkdir(model_dir_base)
        print("Create log dir: %s" %(model_dir_base))

    model_dir = os.path.join(model_dir_base, model_settings)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print("Create log dir: %s" %(model_dir))
    summaryWriter = SummaryWriter(log_dir=log_dir)

    # train_data_dir = "/data1/huangbin/mimic-ii/align_filter_train_manual/"
    # test_data_dir = "/data1/huangbin/mimic-ii/align_filter_test_manual/"

    # test_npz_data = "/data1/huangbin/mimic-ii/test_dataset_SL256_numPeaks%s.npz" % (num_peaks)
    # train_npz_data = "/data1/huangbin/mimic-ii/train_dataset_SL256_numPeaks%s.npz" % (num_peaks)

    test_npz_data = "/data1/huangbin/mimic-ii/test_dataset_numPeaks%s.npz" % (num_peaks)
    train_npz_data = "/data1/huangbin/mimic-ii/train_dataset_numPeaks%s.npz" % (num_peaks)

    train_ecg_list, train_ppg_list, train_BP_list = read_npz_data(train_npz_data)
    test_ecg_list, test_ppg_list, test_BP_list = read_npz_data(test_npz_data)

    ecg_all = np.vstack((train_ecg_list, test_ecg_list))
    ppg_all = np.vstack((train_ppg_list, test_ppg_list))
    BP_all = np.vstack((train_BP_list, test_BP_list))

    random.seed(25671)
    data_zip = list(zip(ecg_all, ppg_all, BP_all))
    random.shuffle(data_zip)
    ecg, ppg, BP = zip(*data_zip)

    ecg_list = list(ecg)
    ppg_list = list(ppg)
    BP_list = list(BP)

    test_count = int(0.15*len(BP_list))
    train_count = int(0.85*len(BP_list))
    test_ecg_list, test_ppg_list, test_BP_list = ecg_list[-test_count:], \
                                                 ppg_list[-test_count:], BP_list[-test_count:]

    train_ecg_list, train_ppg_list, train_BP_list = ecg_list[:train_count], \
                                                 ppg_list[:train_count], BP_list[:train_count]

    test_data = MIMIC_II_Rpeaks_align_v2(test_ecg_list, test_ppg_list, test_BP_list)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=test_batch_size,
                             num_workers=2,
                             shuffle=True)

    model = MLPMixer(in_channels=in_channel, num_patch=num_patches, num_classes=num_classes,
                     dim=dim, depth=depth, token_dim=token_dim,
                     channel_dim=channel_dim, dropout=dropout)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    model = model.double()
    model = model.to(device)

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    d_model = 64
    warm_steps = 2000
    epoch = 200
    optimizer = ScheduledOptim(Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                    betas=(0.9, 0.98), eps=1e-09), d_model, warm_steps)

    num_test_data = test_data.__len__()

    for epoch_i in range(epoch):

        train_data = MIMIC_II_Rpeaks_align_noise_v1(train_ecg_list, train_ppg_list, train_BP_list)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
        num_train_data = train_data.__len__()
        print("num training data: ", num_train_data)

        print('[ Epoch: %d / %d ]' % (epoch_i, epoch))

        train_loss = train_epoch(train_loader, device, model, optimizer, num_train_data)
        summaryWriter.add_scalar("train loss", train_loss, epoch_i)

        current_lr = optimizer._optimizer.param_groups[0]['lr']
        print("current LR: ", current_lr)
        summaryWriter.add_scalar("LR", current_lr, epoch_i)

        # todo: validation
        all_pred_sys_BP, all_sys_BP, test_loss = eval_epoch(test_loader, device, model, total_num=num_test_data)
        summaryWriter.add_scalar("test_loss", test_loss, epoch_i)
        all_sys_BP = 100.0 * np.array(all_sys_BP)
        all_pred_sys_BP = 100.0 * np.array(all_pred_sys_BP)
        metrics_valid = MetricRegression(all_sys_BP, all_pred_sys_BP)
        mae = metrics_valid.MAE()
        std = metrics_valid.STD()

        print("Train loss: %2.6f, Test loss: %2.6f" % (train_loss, test_loss))
        print("Val MAE: %.4f, Val STD: %.4f" % (mae, std))
        summaryWriter.add_scalar("MAE", mae, epoch_i)
        summaryWriter.add_scalar("STD", std, epoch_i)

        length_test_res = len(all_pred_sys_BP)
        num_print = 6
        step = np.floor(length_test_res / num_print)
        # print("### test result ### ")
        # temp_pred = []
        # temp_real = []
        # for i_res in range(num_print):
        #     index = int(i_res * step)
        #     temp_real.append(all_sys_BP[index])
        #     temp_pred.append(all_pred_sys_BP[index])
        #
        # print("real sys BP: ", temp_real)
        # print("pred sys BP: ", temp_pred)

        if (epoch_i + 1) % model_save_epoch == 0:
            model_name = "model_ep" + str(epoch_i+1).zfill(3) + ".pt"
            path_model = os.path.join(model_dir, model_name)
            torch.save(model, path_model)
            print("Saved model: ", path_model)


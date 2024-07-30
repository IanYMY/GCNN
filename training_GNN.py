import yaml
import sys
import os

# read out config(.yml)
configfile = sys.argv[1]
with open(configfile, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

# assign gpu
os.environ['CUDA_VISIBLE_DEVICES'] = config['training_GNN']['cuda']

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import random
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from GNN_model import GNN
from utils import GNN_Dataset

# print without omission
np.set_printoptions(threshold=np.inf)


def train_GNN(training_csv, validation_csv, node_hdf, edge_ind_hdf, edge_attr_hdf, checkpoint_path,
              best_checkpoint_path, log_path, load_checkpoint_path=None, best_previous_checkpoint=None):
    """
    Inputs:
    1) training_data: training hdf file name
    2) validation_data: validation hdf file name
    3) checkpoint_path: path to save checkpoint_path.pt
    4) best_checkpoint_path: path to save best_checkpoint_path.pt
    5) load_checkpoint_path: path to checkpoint file to load; default is None, i.e. training from scratch
    6) best_previous_checkpoint: path to the best checkpoint from the previous round of training (required); default is None, i.e. training from scratch
    Output:
    1) checkpoint file, to load into testing function; saved as: checkpoint_path
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define hyperparameters
    seed = config['training_GNN']['seed']
    epochs = config['training_GNN']['epochs']  # number of training epochs
    batch_size = config['training_GNN']['batch_size']  # batch size to use for training
    learning_rate = config['training_GNN']['learning_rate']  # [0.1,0.01,0.001,0.0001]
    heads = config['training_GNN']['heads']
    ratio = config['training_GNN']['ratio']

    # print hyperparameters
    print('Hyperparameters')
    print('seed = %d' % seed)
    print('epochs = %d' % epochs)
    print('batch size = %d' % batch_size)
    print('learning rate = %f' % learning_rate)




    # seed all random number generators and set cudnn settings for deterministic
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = '0'

    def worker_init_fn(worker_id):
        np.random.seed(int(seed))

    # initialize checkpoint parameters
    checkpoint_epoch = 0
    checkpoint_step = 0
    epoch_train_losses, epoch_val_losses, epoch_avg_corr = [], [], []
    best_rmse = float('inf')

    # define function to return checkpoint dictionary
    def checkpoint_model(model, dataloader, epoch, step):
        validate_dict = validate(model, dataloader)
        model.train()  # ?
        checkpoint_dict = {'model_state_dict': model.state_dict(), 'step': step, 'epoch': epoch,
                           'validate_dict': validate_dict,
                           'epoch_train_losses': epoch_train_losses, 'epoch_val_losses': epoch_val_losses,
                           'epoch_pearsonr': pearsonr[0], 'epoch_rmse': mse ** (1/2), 'best_rmse': best_rmse,
                           'y_true': y_true, 'y_pred': y_pred}
        torch.save(checkpoint_dict, checkpoint_path)
        return checkpoint_dict

    # define function to perform validation
    def validate(model, validation_dataloader):
        # initialize
        model.eval()
        y_true = np.zeros((len(validation_dataset),), dtype=np.float32)
        y_pred = np.zeros((len(validation_dataset),), dtype=np.float32)
        # validation
        for batch_ind, batch_data in enumerate(validation_dataloader):
            batch_data = batch_data.to(device)
            y_ = model(batch_data)
            y = batch_data.y
            y_true[batch_ind * batch_size:batch_ind * batch_size + batch_size] = y.cpu().float().data.numpy()[:, 0]
            y_pred[batch_ind * batch_size:batch_ind * batch_size + batch_size] = y_.cpu().float().data.numpy()[:, 0]
            loss = criterion(y.float(), y_.float())
            print('[%d/%d-%d/%d] validation loss: %.3f' % (epoch + 1, epochs, batch_ind + 1, len(validation_dataset) //
                                                           batch_size, loss))

        # compute r^2
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        # compute mae
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        # compute mse
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        # compute pearson correlation coefficient
        pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))[0]
        # compute spearman correlation coefficient
        spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))[0]
        # write out metrics
        print('r2: {}\tmae: {}\trmse: {}\tpearsonr: {}\t spearmanr: {}'.format(r2, mae, mse ** (1 / 2), pearsonr,
                                                                               spearmanr))
        epoch_val_losses.append(mse)
        epoch_avg_corr.append((pearsonr + spearmanr) / 2)
        model.train()
        return {'r2': r2, 'mse': mse, 'mae': mae, 'pearsonr': pearsonr, 'spearmanr': spearmanr,
                'y_true': y_true, 'y_pred': y_pred, 'best_rmse': best_rmse}

    # construct model
    model = GNN(heads=heads, ratio=ratio, concat=False, edge_dim=8)

    training_dataset = GNN_Dataset(training_csv, node_hdf, edge_attr_hdf, edge_ind_hdf)
    validation_dataset = GNN_Dataset(validation_csv, node_hdf, edge_attr_hdf, edge_ind_hdf)

    # print an example input
    ex_input = training_dataset.__getitem__(15)
    ex_x = np.array(ex_input.x[0:5])
    ex_index = np.array(ex_input.edge_index[:, 0:10])
    ex_attr = np.array(ex_input.edge_attr[0:5])
    ex_y = np.array(ex_input.y)
    print('An example input')
    print('x:')
    print(ex_x)
    print('edge index:')
    print(ex_index)
    print('edge attribute:')
    print(ex_attr)
    print('y:')
    print(ex_y)
    print('----------------------------------------------')

    # construct training and validation dataloaders to be fed to model
    batch_count = len(training_dataset)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True,
                                         worker_init_fn=worker_init_fn, drop_last=False, pin_memory=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                           worker_init_fn=worker_init_fn, drop_last=False, pin_memory=True, num_workers=4)

    # load checkpoint file
    if load_checkpoint_path != None:
        if torch.cuda.is_available():
            model_train_dict = torch.load(load_checkpoint_path)
            best_checkpoint = torch.load(best_previous_checkpoint)
        else:
            model_train_dict = torch.load(load_checkpoint_path, map_location=torch.device('cpu'))
            best_checkpoint = torch.load(best_previous_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(model_train_dict['model_state_dict'], strict=True)
        checkpoint_epoch = model_train_dict['epoch']
        checkpoint_step = model_train_dict['step']
        epoch_train_losses = model_train_dict['epoch_train_losses']
        epoch_val_losses = model_train_dict['epoch_val_losses']
        epoch_avg_corr = model_train_dict['epoch_avg_corr']
        val_dict = model_train_dict['validate_dict']
        torch.save(best_checkpoint, best_checkpoint_path)
        best_rmse = best_checkpoint["best_rmse"]

    model.train()
    model.to(device)

    # set loss as MSE
    criterion = nn.MSELoss().float()

    # set Adam optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # train model
    step = checkpoint_step
    for epoch in range(checkpoint_epoch, epochs):
        y_true = np.zeros((len(training_dataset),), dtype=np.float32)
        y_pred = np.zeros((len(training_dataset),), dtype=np.float32)
        for batch_ind, batch_data in enumerate(training_dataloader):
            batch_data = batch_data.to(device)
            y_ = model(batch_data)
            y = batch_data.y
            y_true[batch_ind * batch_size:batch_ind * batch_size + batch_size] = y.cpu().float().data.numpy()[:, 0]
            y_pred[batch_ind * batch_size:batch_ind * batch_size + batch_size] = y_.cpu().float().data.numpy()[:, 0]

            # compute loss and update parameters
            loss = criterion(y.float(), y_.float())
            loss.backward()
            optimizer.step()
            step += 1
            print("[%d/%d-%d/%d] training loss: %.3f" % (
            epoch + 1, epochs, batch_ind + 1, len(training_dataset) // batch_size, loss))

        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        mse = mean_squared_error(y_true, y_pred)
        epoch_train_losses.append(mse)
        pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
        spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))

        # write training summary for the epoch
        print('epoch: {}\trmse:{:0.4f}\tr2: {:0.4f}\t pearsonr: {:0.4f}\tspearmanr: {:0.4f}\tmae: {:0.4f}\tpred'.format(
            epoch + 1, mse ** (1 / 2), r2, float(pearsonr[0]),
            float(spearmanr[0]), float(mae)))

        checkpoint_dict = checkpoint_model(model, validation_dataloader, epoch + 1, step)
        if (checkpoint_dict["validate_dict"]["mse"] ** (1/2)) < best_rmse:
            best_rmse = checkpoint_dict["validate_dict"]["mse"] ** (1/2)
            torch.save(checkpoint_dict, best_checkpoint_path)
        torch.save(checkpoint_dict, checkpoint_path)

    # print best epoch
    if torch.cuda.is_available():
        best_dict = torch.load(best_checkpoint_path)
    else:
        best_dict = torch.load(best_checkpoint_path, map_location=torch.device('cpu'))

    best_epoch = best_dict['epoch']
    print('best epoch: %d' % best_epoch)

    # learning curve and correlation plot
    fig, axs = plt.subplots(2, figsize=[8, 6])
    axs[0].axvline(x=best_epoch, c="r", ls="--")
    axs[0].text(0, 9, ' best epoch='+str(best_epoch), fontsize=8, verticalalignment='top', horizontalalignment='left')
    axs[0].plot(np.arange(1, epochs + 1), np.array(epoch_train_losses), label='training')
    axs[0].plot(np.arange(1, epochs + 1), np.array(epoch_val_losses), label='validation')
    axs[0].set_xlabel('Epoch', fontsize=10)
    axs[0].set_ylabel('Loss', fontsize=10)
    axs[0].set_ylim(0, 10)
    axs[0].legend(fontsize=9)
    axs[1].plot(np.arange(1, epochs + 1), np.array(epoch_avg_corr))
    axs[1].set_xlabel('Epoch', fontsize=10)
    axs[1].set_ylabel('Validation Correlation', fontsize=10)
    axs[1].set_ylim(0, 1)
    plt.savefig(os.path.join(log_path, 'epoch_loss.png'), dpi=300)

    loss_data = {'epoch training losses': np.array(epoch_train_losses), 'epoch validation losses': np.array(epoch_val_losses)}
    df = pd.DataFrame(loss_data)
    df.to_csv(os.path.join(log_path, 'epoch_loss.csv'), index=False)


training_csv = config['training_GNN']['training_csv']
validation_csv = config['training_GNN']['validation_csv']
node_hdf = config['training_GNN']['node_hdf']
edge_ind_hdf = config['training_GNN']['edge_ind_hdf']
edge_attr_hdf = config['training_GNN']['edge_attr_hdf']
checkpoint_path = config['training_GNN']['checkpoint_path']
best_checkpoint_path = config['training_GNN']['best_checkpoint_path']
log_path = config['training_GNN']['log_path']

if __name__ == "__main__":
    train_GNN(training_csv, validation_csv, node_hdf, edge_ind_hdf, edge_attr_hdf, checkpoint_path,
              best_checkpoint_path, log_path, load_checkpoint_path=None, best_previous_checkpoint=None)


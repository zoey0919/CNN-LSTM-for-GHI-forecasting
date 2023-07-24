import os
import random
import torch
import pandas
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from ife_lstm import *
import matplotlib.pyplot as plt
from torchvision import transforms
from generateimg import *
import time
from torch.optim import lr_scheduler
from earlystopping import EarlyStopping
from tqdm import tqdm

T1 = time.time()
if __name__ == '__main__':
    TIMESTAMP = '21'

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    def set_random_seed(seed, deterministic=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    set_random_seed(1, deterministic=False)
    learning_rate = 0.001
    epochs = 200
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((128, 128))
    ]
    data_transforms = transforms.Compose(transform_list)
    train_data = SeqDataset(txt='/home/lius/solarforecast/data/train_2path.txt', xlsx='/home/lius/solarforecast/data/train_2in.xlsx', transform=data_transforms)
    batch_size = 50
    valid_data = SeqDataset(txt='/home/lius/solarforecast/data/valid_2path.txt', xlsx='/home/lius/solarforecast/data/valid_2in.xlsx', transform=data_transforms)
    train_loader = DataLoader(train_data, shuffle=True, num_workers=10, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, num_workers=10, batch_size=batch_size)
    ###加载模型
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=30, verbose=True)
    
    model = IFE()
    
    model.cuda()
    model_save_path = 'save_model/' + TIMESTAMP
    if os.path.exists(os.path.join(model_save_path, 'Checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        loaded_paras = torch.load(os.path.join(model_save_path, 'Checkpoint.pth.tar'))
        model.load_state_dict(loaded_paras)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(loaded_paras['optimizer'])
        cur_epoch = loaded_paras['epoch'] + 1
    else:
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)
        cur_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)  # 定义优化器
    loss_func = nn.MSELoss()
    loss_func = loss_func.cuda()
    # tune the learning_rate
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.1,
                                                      patience=5,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    for epoch in range(cur_epoch, epochs + 1):
        t = tqdm(train_loader, leave=False, total=len(train_loader))
        for i, (inputVar, targetVar) in enumerate(t):
            inputs = inputVar.cuda()  # B,S,C,H,W
            label = targetVar.cuda()  # B,S,C,H,W
            optimizer.zero_grad()
            model.train()
            pred = model(inputs)  # B,S,C,H,W
            loss = loss_func(pred, label)
            train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss.item()),
                'epoch': '{:02d}'.format(epoch)
            })
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            model.eval()
            t = tqdm(valid_loader, leave=False, total=len(valid_loader))
            for i, (inputVar, targetVar) in enumerate(t):
                inputs = inputVar.cuda()
                label = targetVar.cuda()
                pred = model(inputs)
                loss = loss_func(pred, label)
                # record validation loss
                valid_losses.append(loss.item())
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss.item()),
                    'epoch': '{:02d}'.format(epoch)
                })
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, model_save_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open(model_save_path+'/'+"avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open(model_save_path+'/'+"avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(avg_train_losses[1:], 'r', label='train_loss')
    plt.plot(avg_valid_losses[1:], 'b', label='valid_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(model_save_path + '/loss.jpg')
    T2 = time.time()
    print('running time:%shour' % ((T2 - T1) / 3600))
    
       

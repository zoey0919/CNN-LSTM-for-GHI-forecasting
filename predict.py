import os
import torch
import numpy as np
import pandas
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from ife_lstm import *
import matplotlib.pyplot as plt
from torchvision import transforms
from generateimg import *
import random
import tqdm
if __name__ == '__main__':
    def set_random_seed(seed, deterministic=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    set_random_seed(seed=1996, deterministic=False)

    model_save_path = 'save_model/3_1/checkpoint_50_0.013140.pth.tar'
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((128, 128))
    ]
    data_transforms = transforms.Compose(transform_list)
    test_data = SeqDataset(txt='test_3path.txt', xlsx='test_3in.xlsx', transform=data_transforms)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=1, batch_size=1)
    loaded_paras = torch.load(model_save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IFE_LSTM().to(device)
    state_dict = loaded_paras['state_dict']
    # state_dict = loaded_paras
    model.load_state_dict(state_dict)
    print("#### 成功载入已有模型，进行预测...")
    ##测试模型
    with torch.no_grad():
        model.eval()
        test_loss = []
        predictions = []
        actuals = []
        loss_func = nn.MSELoss()
        t = tqdm.tqdm(test_loader, leave=False, total=len(test_loader))
        for i, (inputVar1, inputVar2, targetVar) in enumerate(t):
            inputs1 = inputVar1.to(device)  # B,S,C,H,W
            inputs2 = inputVar2.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            pred = model(inputs1, inputs2)
            loss = loss_func(pred, label)
            test_loss.append(loss.item())
            predictions.append(pred)
            actuals.append(label)
        losss = np.mean(test_loss)
        print(losss)
        predictions = torch.tensor(predictions).cuda().data.cpu().numpy()
        actuals = torch.tensor(actuals).cuda().data.cpu().numpy()
        data1 = pandas.DataFrame(actuals)
        data = pandas.DataFrame(predictions)
        data = pandas.concat([data1, data], axis=1)
        data.to_excel('save_model/3_1/test_0.0152734.xlsx', index=False)

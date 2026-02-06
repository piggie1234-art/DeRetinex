import torch
import torch.nn.functional as F
import torchvision.models as models
# import sys
# print(sys.path)
from Net import *
from Loss import *
from dataloader import *
from torch.utils.data import DataLoader
import time
#from torch.nn.functional import cosine_similarity
from torch.optim.lr_scheduler import StepLR
from torchvision import models
import torch.nn.functional as F
from utils import calculate_al_mask


train_batch_size = 16
start_epochs = 0
learning_rate = 0.0002

num_epochs = 400
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_point = 20

model_choose = 1



consLoss = nn.MSELoss()
recLoss = nn.MSELoss()
colorLoss = nn.MSELoss()
hazeLoss = nn.MSELoss() 
# # structure-aware TV loss
# smoothLoss = TVLoss()
l1_loss = nn.L1Loss()
ssimloss = SSIMLoss()
#smoothLoss = SmoothnessLoss()
smoothLoss = IlluminationSmoothnessLoss()

class DataPrefetcher():

    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

    def next(self):
        batch = self.batch
        self.preload()
        return batch
    
def train_1(start_epoch):
    model_1 = DecomNet_DecompositionNet_L().to(device)
    model1_name = model_1.__class__.__name__
    print(f"导入模型{model1_name}")
    if start_epoch != 0:
        model1_path = f'/home/zhw/UIALN/saved_model/saved_model1/epoch_{epoch}.pth'
        model_1.load_state_dict(torch.load(model1_path)['model'])
    print("模型导入完成")
    model_1.train()
    optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate,weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    total_loss = 0
    L_no_light_path = r"/home/zhw/UIALN/Synthetic_dataset/synthetic_dataset_no_AL"
    L_light_path = r"/home/zhw/UIALN/Synthetic_dataset/synthetic_dataset_with_AL/train"
    dataset = retinex_decomposition_data(L_no_light_path, L_light_path)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    for epoch in range(start_epoch+1, num_epochs+1):
        print("epoch: ", epoch)
        start_time = time.time()
        prefetcher = DataPrefetcher(train_loader)
        batch = prefetcher.next()
        i = 0
        epoch_loss = 0
        while batch is not None:
            i += 1
            L_no_light = batch[0].to(device)
            L_light = batch[1].to(device)
            I_no_light_hat, R_no_light_hat = model_1(L_no_light)
            I_light_hat, R_light_hat = model_1(L_light)
            loss_1 = l1_loss(R_light_hat, R_no_light_hat)
            loss_2 = l1_loss(I_light_hat*R_light_hat, L_light) + l1_loss(I_no_light_hat*R_no_light_hat, L_no_light) + 0.001*l1_loss(I_no_light_hat*R_no_light_hat, L_light) + 0.001*l1_loss(I_light_hat*R_light_hat, L_no_light)
            loss_3 = smoothLoss(I_light_hat, R_light_hat) + smoothLoss(I_no_light_hat, R_no_light_hat)
            loss = loss_1 + 0.001*loss_2 + 0.1*loss_3
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch = prefetcher.next()
        if epoch % save_point == 0:
            state = {'model': model_1.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, f'/home/zhw/UIALN/saved_model/saved_model1/epoch_{epoch}.pth')
        current_lr = optimizer.param_groups[0]['lr'] #获取当前学习率
        scheduler.step()
        time_epoch = time.time() - start_time
        epoch_loss = epoch_loss*1.0/i
        total_loss += epoch_loss
        print("==>Now: epoch:{}/{}, time: {:.2f} min , loss: {:.5f},current_lr:{}".format(epoch,num_epochs, time_epoch / 60, epoch_loss,current_lr))
        if epoch == start_epochs + 1:
            with open("output.txt", "a") as f:
                f.write("train_1 information\n")
        with open("output.txt", "a") as f:
            f.write("==>No: epoch:{}/{}, time: {:.2f} min, loss: {:.5f},current_lr:{}\n".format(epoch,num_epochs, time_epoch / 60, epoch_loss,current_lr))
    print("total_loss:",total_loss*1.0/num_epochs-start_epochs)


def train_2(start_epoch):
    model_2 = Illumination_Correction().to(device)
    print(f"导入模型{model_2.__class__.__name__}")
    
    if start_epoch != 0:
        model2_path = f'UIALN/saved_model/saved_model2/excellent/epoch_90.pth'
        model_2.load_state_dict(torch.load(model2_path)['model'])
    
    print("模型导入完成")
    model_2.train()
    optimizer = torch.optim.Adam(model_2.parameters(), lr=learning_rate,weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.5)

    total_loss = 0
    L_no_light_path = r"/home/zhw/UIALN_copy/NoAl_retinex"
    L_light_path = r"/home/zhw/UIALN_copy/Al_retinex/train"
    dataset = IlluminationDataset(L_no_light_path, L_light_path)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    for epoch in range(start_epoch+1, num_epochs+1):
        start_time = time.time()
        prefetcher = DataPrefetcher(train_loader)
        batch = prefetcher.next()
        i = 0
        epoch_loss = 0
        while batch is not None:
            i+=1

            I_no_light,I_light,R_no_light,R_light =  batch['noal_illum'].to(device),batch['al_illum'].to(device),batch['noal_refl'].to(device),batch['al_refl'].to(device)
            I_delight_hat = model_2(torch.cat((I_light, R_light), dim=1))

            loss_1 = consLoss(I_delight_hat*R_light, I_no_light*R_no_light) 
            loss_2 = recLoss(R_no_light,R_light)
            loss_3 = l1_loss(I_delight_hat, I_no_light) 
            loss_4 = ssimloss(I_delight_hat, I_no_light)
            loss = loss_1  + loss_2 + loss_3 + loss_4 

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch = prefetcher.next()
        if epoch % save_point == 0:
            state = {'model': model_2.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, f'/home/zhw/UIALN/saved_model/saved_model2/epoch_{epoch}.pth')
        current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
        scheduler.step()
        time_epoch = time.time() - start_time
        epoch_loss = epoch_loss/i
        total_loss += epoch_loss
        print("==>Now: epoch:{}/{}, time: {:.2f} min , loss: {:.5f},current_lr:{}".format(epoch,num_epochs, time_epoch / 60, epoch_loss,current_lr))
        if epoch == start_epochs + 1:
            with open("output.txt", "a") as f:
                f.write("train_2 information\n")
        with open("output.txt", "a") as f:
            f.write("==>No: epoch:{}/{}, time: {:.2f} min, loss: {:.5f},current_lr:{}\n".format(epoch,num_epochs, time_epoch / 60, epoch_loss,current_lr))
    print("total_loss:",total_loss*1.0/(num_epochs-start_epochs))

def train_3(start_epoch):
    print("模型导入")
    # 前置模型
    model_2 = Illumination_Correction().to(device)
    model2_path = '/home/zhw/UIALN/saved_model/saved_model2/excellent/epoch_400.pth'
    model_2.load_state_dict(torch.load(model2_path)['model'])
    # 后置模型
    model_3 = AL_Area_Selfguidance_Color_Correction().to(device)
    if start_epoch != 0:
        model3_path = '/home/zhw/UIALN/saved_model/saved_model3/epoch_' + str(start_epoch) + '.pth'
        model_3.load_state_dict(torch.load(model3_path)['model'])
    print("模型导入完成")
    model_2.eval()
    model_3.train()
    I_light_path = r"/home/zhw/UIALN_copy/Al_retinex/train"
    ABcc_path = r"/home/zhw/UIALN/Synthetic_dataset/synthetic_dataset_with_AL/train"
    gt_path = r"/home/zhw/UIALN/Synthetic_dataset/labels/raw"
    dataset = AL_data(I_light_path,ABcc_path, gt_path)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model_3.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    total_loss = 0
    for epoch in range(start_epoch+1, num_epochs+1):
        print("epoch: ", epoch)
        start_time = time.time()
        prefetcher = DataPrefetcher(train_loader)
        batch = prefetcher.next()
        i = 0
        epoch_loss = 0
        while batch is not None:
            i+=1
            if i%50 == 0:
                print("batch: ", i)
            I_light ,ABcc, gt,R_light= batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            I_delight = model_2(torch.cat((I_light, R_light), dim=1))
            M_image = calculate_al_mask(I_light,I_delight)
            ABcc_hat = model_3(M_image, ABcc)
            loss = colorLoss(ABcc_hat, gt) #+ nn.L1Loss()(ABcc_hat, gt) 
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch = prefetcher.next()
        if epoch % save_point == 0:
            state = {'model': model_3.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, '/home/zhw/UIALN/saved_model/saved_model3/epoch_' + str(epoch) + '.pth')
        time_epoch = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
        scheduler.step()
        epoch_loss = epoch_loss*1.0/i
        total_loss += epoch_loss
        print("==>Now: epoch:{}/{}, time: {:.2f} min, loss: {:.5f},current_lr:{}".format(epoch,num_epochs, time_epoch / 60, epoch_loss,current_lr))
        if epoch == start_epochs + 1:
            with open("output.txt", "a") as f:
                f.write("train_3 information\n")
        with open("output.txt", "a") as f:
            f.write("==>Now: epoch:{}/{}, time: {:.2f} min, loss: {:.5f},current_lr:{}\n".format(epoch,num_epochs, time_epoch / 60, epoch_loss,current_lr))
    print("total_loss:",total_loss/(num_epochs-start_epochs))

def train_4(start_epoch):
    print("模型导入")
    # 前置模型
    model_2 = Illumination_Correction().to(device)
    model2_path = '/home/zhw/UIALN/saved_model/saved_model2/excellent/epoch_400.pth'
    model_2.load_state_dict(torch.load(model2_path)['model'])
    model_3 = AL_Area_Selfguidance_Color_Correction().to(device)
    model3_path = '/home/zhw/UIALN/saved_model/saved_model3/excellent/synthetic/epoch_240.pth'
    model_3.load_state_dict(torch.load(model3_path)['model'])
    I_light_path = r"/home/zhw/UIALN_copy/Al_retinex/train"
    ABcc_path = r"/home/zhw/UIALN/Synthetic_dataset/synthetic_dataset_with_AL/train"
    gt_path = r"/home/zhw/UIALN/Synthetic_dataset/labels/raw"
    dataset = Detail_Enhancement_data(I_light_path,ABcc_path, gt_path)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    # 后置模型
    model_4 = Detail_Enhancement().to(device)
    model_fusion = Channels_Fusion_with_FCANet().to(device)
    if start_epoch != 0:
        model4_path = '/home/zhw/UIALN/saved_model/saved_model4/epoch_' + str(start_epoch) + '.pth'
        model_4.load_state_dict(torch.load(model4_path)['model'])
        model_fusion_path = '/home/zhw/UIALN/saved_model/saved_modelFusion/epoch_' + str(start_epoch) + '.pth'
        model_fusion.load_state_dict(torch.load(model_fusion_path)['model'])
    print("模型导入完成")

    model_2.eval()
    model_3.eval()
    model_4.train()
    model_fusion.train()
    
    optimizer_4 = torch.optim.Adam(model_4.parameters(), lr=learning_rate)
    optimizer_fusion = torch.optim.Adam(model_fusion.parameters(), lr=learning_rate)
    optimizer = [optimizer_4, optimizer_fusion]
    scheduler = [
        StepLR(optimizer_4, step_size=20, gamma=0.5),
        StepLR(optimizer_fusion, step_size=20, gamma=0.5)
        ]

    total_loss = 0
    for epoch in range(start_epoch+1, num_epochs+1):
        print("epoch: ", epoch)
        start_time = time.time()
        prefetcher = DataPrefetcher(train_loader)
        batch = prefetcher.next()
        i = 0
        epoch_loss = 0
        while batch is not None:
            i+=1
            if i%50 == 0:
                print("batch: ", i)
            I_light ,ABcc , GT_lab , R_light , GT_l = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)

            I_delight = model_2(torch.cat((I_light, R_light), dim=1))
            M_image = calculate_al_mask(I_light,I_delight)
            ABcc = model_3(M_image, ABcc)
            L_delight = I_delight * R_light
            
            L_en_hat = model_4(L_delight)   # enhanced L
            LAB_hat = torch.cat((L_en_hat, ABcc), dim=1)
            LAB_hat = model_fusion(LAB_hat)
            
            loss_haze = hazeLoss(GT_l, L_en_hat)
            loss_ssim = ssimloss(L_en_hat, GT_l)
            # print('预测的L范围:',L_en_hat.min(), L_en_hat.max())
            # print('真实的L范围:',GT_l.min(), GT_l.max())
            loss_recons = recLoss(GT_lab, LAB_hat)
            final_loss = loss_haze + loss_recons + loss_ssim
            epoch_loss += final_loss.item()
            
            optimizer_fusion.zero_grad()
            optimizer_4.zero_grad()
            final_loss.backward()
            optimizer_fusion.step()
            
            
            # final_loss.backward()
            optimizer_4.step()                
            
            batch = prefetcher.next()
        if epoch % save_point == 0:
            state = {'model': model_4.state_dict(), 'optimizer': optimizer_4.state_dict(), 'epoch': epoch}
            torch.save(state, '/home/zhw/UIALN/saved_model/saved_model4/epoch_' + str(epoch) + '.pth')
            state = {'model': model_fusion.state_dict(), 'optimizer': optimizer_fusion.state_dict(), 'epoch': epoch}
            torch.save(state, '/home/zhw/UIALN/saved_model/saved_modelFusion/epoch_' + str(epoch) + '.pth')
        time_epoch = time.time() - start_time
        for scheduler_ in scheduler:
            scheduler_.step()
        current_lrofm4 = optimizer[0].param_groups[0]['lr']
        current_lrofmFusion = optimizer[1].param_groups[0]['lr']
        epoch_loss = epoch_loss*1.0/i
        total_loss += epoch_loss
        print("==>Now: epoch:{}/{}, time: {:.2f} min, loss: {:.5f},current_lrof4:{},current_lrofmFusion:{}".format(epoch,num_epochs, time_epoch / 60, epoch_loss,current_lrofm4,current_lrofmFusion))
        if epoch == start_epochs + 1:
            with open("output.txt", "a") as f:
                f.write("train_last information\n")
        with open("output.txt", "a") as f:
            f.write("==>Now: epoch:{}/{}, time: {:.2f} min , loss: {:.5f},current_lrof4:{},current_lrofmFusion:{}\n".format(epoch,num_epochs, time_epoch / 60, epoch_loss,current_lrofm4,current_lrofmFusion))
    print("total_loss:",total_loss*1.0/(num_epochs-start_epochs))


if __name__ == '__main__':
    print(torch.cuda.is_available())
    if model_choose == 1:
        train_1(start_epochs)
    if model_choose == 2:
        train_2(start_epochs)
    elif model_choose == 3:
        train_3(start_epochs)
    elif model_choose == 4:
        train_4(start_epochs)
    else:
        print("model_choose error")
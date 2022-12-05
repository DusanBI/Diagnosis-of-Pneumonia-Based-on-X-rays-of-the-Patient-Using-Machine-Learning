import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import copy
import pickle
import time
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore') 
    
    
def train_model(model, dataloaders, criterion, optimizer,
                model_name, pretrained, num_epochs=1):
    print('\n... Training of {} ...\n'.format(model_name))
    since = time.time()
    
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    val_prec_history = []
    val_rec_history = []
    val_f1_history = []
    val_tn_history = []
    val_fp_history = []
    val_fn_history = []
    val_tp_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            epoch_preds = []
            epoch_labels = []
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # loss.item() is an average loss over the batch, so we multiply it by batch_size
                running_loss += loss.item() * inputs.size(0)
                epoch_preds.append(preds.detach().cpu().numpy())
                epoch_labels.append(labels.detach().cpu().numpy())
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)    
                
            epoch_preds = np.concatenate(epoch_preds)
            epoch_labels = np.concatenate(epoch_labels)
            
            epoch_acc = accuracy_score(epoch_labels, epoch_preds)
            epoch_rec = precision_score(epoch_labels, epoch_preds)
            epoch_prec = recall_score(epoch_labels, epoch_preds)
            epoch_f1 = f1_score(epoch_labels, epoch_preds)
            epoch_tn, epoch_fp, epoch_fn, epoch_tp = confusion_matrix(epoch_labels, epoch_preds,
                                                                      labels=[True, False]).ravel()
            
            print('{} Loss: {:.4f} Acc: {:.4f} \t Prec: {:.4f} \t Rec: {:.4f} \t F1: {:.4f}'\
                  .format(phase, epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1))

            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_f1 = epoch_f1
                best_acc = epoch_acc
                best_prec = epoch_prec
                best_rec = epoch_rec
                best_tn = epoch_tn
                best_fp = epoch_fp
                best_fn  = epoch_fn
                best_tp = epoch_tp

            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                val_prec_history.append(epoch_prec)
                val_rec_history.append(epoch_rec)
                val_f1_history.append(epoch_f1)
                val_tn_history.append(epoch_tn)
                val_fp_history.append(epoch_fp)
                val_fn_history.append(epoch_fn)
                val_tp_history.append(epoch_tp)
            else:
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    result_dict = {'model': model_name,
                   'val_acc_history': val_acc_history,
                   'val_prec_history': val_prec_history,
                   'val_rec_history': val_rec_history,
                   'val_f1_history': val_f1_history,
                   'val_tn_history': val_tn_history,
                   'val_fp_history': val_fp_history,
                   'val_fn_history': val_fn_history, 
                   'val_tp_history': val_tp_history,
                   'best_f1': best_f1,
                   'best_acc': best_acc,
                   'best_prec': best_prec, 
                   'best_rec': best_rec,
                   'best_tn': best_tn, 
                   'best_fp': best_fp, 
                   'best_fn': best_fn,
                   'best_tp': best_tp}
    if pretrained:
        with open('../results/{}_pretrained_res.pkl'.format(model_name), 'wb') as outfile:
            pickle.dump(result_dict, outfile)
    else:
        with open('../results/{}_not_pretrained_res.pkl'.format(model_name), 'wb') as outfile:
            pickle.dump(result_dict, outfile)
    return model
    

class PneumoniaDataset(Dataset):
    def __init__(self, dataset_df, transforms=None): 
        self.dataset_df = dataset_df
        self.transforms = transforms
        
    def __getitem__(self, index):
        image_label = self.dataset_df.iloc[index]['label']
        img_path = self.dataset_df.iloc[index]['path']
        img_array = Image.open(img_path).convert('RGB')
        
        if self.transforms is not None:
            img_array = self.transforms(img_array)
        
        return img_array, image_label
    
    def __len__(self):
        return len(self.dataset_df)

    
class Args():
    IM_H = 224
    IM_W = 224
    BATCH_SIZE = 16
    MODELS = ['alexnet', 'densenet', 'vgg16', 'resnet']
    PRETRAINED = False
    LAST_ONLY = True
    EPOCHS = 30
    LR = 0.001

    
if __name__ == '__main__':
    
    args = Args()
    paths = []
    sets = []
    classes = []

    for root, dirs, files in os.walk(os.path.abspath("../data/")):
        for file in files:
            if os.path.join(root, file).endswith('jpeg'):
                abs_path = os.path.join(root, file)
                paths.append(abs_path)
                sets.append(abs_path.split('/')[5])
                classes.append(abs_path.split('/')[6])
    
    paths_df = pd.DataFrame({
        'path': paths,
        'set': sets,
        'label': [0 if x == 'normal' else 1 for x in classes]
    })
    print(paths_df.head())
    print(len(paths_df))
    train_transforms = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.RandomRotation((-20, 20)),
        T.RandomAffine(
            0, 
            translate=None,
            scale=[0.7, 1.3],
            shear=None,
            resample=False,
            fillcolor=0),
        T.Resize(size=(args.IM_H, args.IM_W)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    val_transforms = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Resize(size=(args.IM_H, args.IM_W)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    
    train_df = paths_df[paths_df.set == 'train']
    val_df = paths_df[paths_df.set == 'val']
    test_df = paths_df[paths_df.set == 'test']           
    
    train_dataset = PneumoniaDataset(train_df, transforms=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)

    val_dataset = PneumoniaDataset(val_df, transforms=train_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    test_dataset = PneumoniaDataset(test_df, transforms=train_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    print("Training dataset contains {} samples...".format(len(train_dataset)))
    print("Validation dataset contains {} samples...".format(len(val_dataset)))
    print("Test dataset contains {} samples...".format(len(test_dataset)))
    
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    
    for model_name in args.MODELS:
        for pretrained in [True, False]:
            device = torch.device("cuda:0")   
            # device = torch.device('cpu')

            if model_name == 'alexnet':
                model = models.alexnet(pretrained=pretrained)
                model.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
                if pretrained and args.LAST_ONLY:
                    for name, param in model.named_parameters():
                        if 'classifier.6' not in name:
                            param.requires_grad=False
            elif model_name == 'densenet':
                model = models.densenet161(pretrained=pretrained)
                model.classifier = nn.Linear(in_features=2208, out_features=2, bias=True)
                if pretrained and args.LAST_ONLY:
                    for name, param in model.named_parameters():
                        if 'classifier' not in name:
                            param.requires_grad=False
            elif model_name == 'resnet':
                model = models.resnet18(pretrained=pretrained)
                model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
                if pretrained and args.LAST_ONLY:
                    for name, param in model.named_parameters():
                        if 'fc' not in name:
                            param.requires_grad=False
            elif model_name == 'vgg16':
                model = models.vgg16(pretrained=pretrained)
                model.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
                if pretrained and args.LAST_ONLY:
                    for name, param in model.named_parameters():
                        if 'classifier.6' not in name:
                            param.requires_grad=False
            else: 
                raise Exception('Chosen model not supported!')
            
            model.to(device)
            
            loss_function = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=0.0005)
            
            best_model = train_model(model, dataloaders, loss_function, optimizer,
                                     model_name, pretrained, num_epochs=args.EPOCHS)
            if pretrained:
                save_path = '../models/' + model_name + '_pretrained'
            else:
                save_path = '../models/' + model_name + '_not_pretrained'
            torch.save(best_model.state_dict(), save_path)

            

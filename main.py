from torchvision import models 
from types import SimpleNamespace 
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import torch 
import torchvision 
import torch.nn as nn 
import torch.utils.data as data 
import torchvision.transforms as transforms
import torchvision.datasets as dsets 


def init_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device


config = SimpleNamespace() 
config.content = 'content.jpg' 
config.style = 'style.jpg' 
config.maxSize = 400 
config.totalStep = 300
config.step = 1
config.sampleStep = 100 
config.lr = .003 
device = init_device()


class PretrainedNet(nn.Module):

        def __init__(self):
        
            super(PretrainedNet, self).__init__()
            self.select = [0, 5, 7, 10, 15] 
            self.pretrainedNet = models.vgg19(pretrained=True).to(device) 
        
        def forward(self, x):

            features = [] 
            output = x
            for layerIndex in range(len(self.pretrainedNet.features)):   
                output = self.pretrainedNet.features[layerIndex](output)   
                if layerIndex in self.select:                              
                    features.append(output)   

            return features 


def load_image(image_path, transform=None, maxSize=None, shape=None):
    
    
    image = Image.open(image_path)
    
    if maxSize:
        scale = maxSize / max(image.size) 
        size = np.array(image.size) * scale 
        image = image.resize(size.astype(int), Image.ANTIALIAS) 
        
    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0) 
    
    return image.to(device)


def train():


    transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
    content = load_image(config.content, transform, maxSize=config.maxSize)
    style = load_image(config.style, transform, shape=[content.size(2), content.size(3)])
    target = content.clone().requires_grad_(True)

    model = PretrainedNet().eval() 

    optimizer = torch.optim.Adam([target], lr=0.1) 
    contentCriteria = nn.MSELoss()
    device = init_device()


    for step in range(config.totalStep):
        
        targetFeatures = model.forward(target)   
        contentFeatures = model.forward(content) 
        styleFeatures = model.forward(style)     
        
        styleLoss = 0    
        contentLoss = 0  

        for f1, f2, f3 in zip(targetFeatures, contentFeatures, styleFeatures):
            
            contentLoss += contentCriteria(f1, f2)

            _, c, h, w = f1.size() 
            f1 = f1.reshape(c, h*w).to(device) 
            f3 = f3.reshape(c, h*w).to(device)

            f1 = torch.mm(f1, f1.t()) 
            f3 = torch.mm(f3, f3.t())

            kf1 = 1 / (4 * (len(f1)*len(f3))**2)
            kf2 = 1 / 4 * (len(f1)*len(f3))**2
            kf3 = 1 / (c * w * h)
            
            styleLoss += contentCriteria(f1,f3) * kf2
        
        loss = styleLoss + contentLoss 
        optimizer.zero_grad()          
        loss.backward()                
        optimizer.step()               


        if (step+1) % config.step == 0: 
            print('Шаг [{}/{}], Ошибка для оригинала: {:.4f}, Ошибка для стиля: {}' 
                .format(step+1, config.totalStep, contentLoss.item(), styleLoss.item()))

        if (step+1) % config.sampleStep == 0: 
            img = target.clone().squeeze()
            img = img.clamp_(0, 1) 
            torchvision.utils.save_image(img, 'output-{}.png'.format(step+1))
    

    def show_result():

        target_img = target.cpu().detach().numpy()[0].transpose(1,2,0)
        plt.figure(figsize=(14,7))
        plt.imshow(target_img)
        plt.axis('off')
        plt.show()

    show_result()


if __name__ == '__main__':
    train()
    


    

    
    

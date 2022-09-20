import torch
from PIL import Image
import numpy as np

class IMDBDataset(torch.utils.data.Dataset):
    """CelebA dataloader, output image and target"""
    
    def __init__(self, image_feature, target_dict, sex_dict, mode, eb1, eb2, test, dev_or_test, transform=None):
        self.image_feature = image_feature
        self.target_dict = target_dict
        self.sex_dict = sex_dict
        self.transform = transform

        if mode == 'eb1':
            self.key_list = eb1
        elif mode == 'eb2':
            self.key_list = eb2
        elif mode == 'unbiased':
            self.key_list = test
        elif mode == 'all':
            self.key_list = eb1 + eb2 + test
        
        if dev_or_test == 'train':
            pass
        elif dev_or_test == 'dev':
            self.key_list = self.key_list[:len(self.key_list) // 10]
        elif dev_or_test == 'test':
            self.key_list = self.key_list[len(self.key_list) // 10:]
        
        # print(len(self.key_list))
        # print('dasjldjlkas')
        # print(target_dict)
        # print(target_dict['nm3924265_rm538246144_1989-2-20_2014.jpg'.encode('ascii')])

    def __getitem__(self, index):
        key = self.key_list[index]
        img = Image.fromarray(self.image_feature[key][()])
        sex = np.array(self.sex_dict[key.encode('ascii')])[np.newaxis]

        if self.transform is not None:
            img = self.transform(img)
        
        return img, torch.FloatTensor(sex)

    def __len__(self):
        return len(self.key_list)
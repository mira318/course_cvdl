import torch
import cv2
from torch.utils import data
from torchvision import transforms

class VideoDataset(data.Dataset):
    def __init__(self, root, index_file, vid_transforms, clip_size = 8, nclips = 1, step_size = 10):
        self.root = root
        self.clip_size = clip_size
        self.step_size = step_size
        
        self.files_list = []
        self.classes_list = []
        self.vid_transforms = vid_transforms
        
        with open(index_file, 'r') as f:
            line = f.readline()
            while line:
                file_loc, file_class = line.split()
                self.files_list.append(file_loc)
                self.classes_list.append(int(file_class))
                line = f.readline()
            
        
    def  __getitem__(self, index):
        
        path = self.root + self.files_list[index]
        item_class = self.classes_list[index]
        
        catcher = cv2.VideoCapture(path)
        success, image = catcher.read()
        images = []
        t = 0
        while success:
            if t % self.step_size == 0:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.vid_transforms(image)
                images.append(image)
            success, image = catcher.read()
            t += 1
            
        duritation = len(images)
        need_at_least = self.clip_size
        
        if need_at_least < duritation:
            len_diff = duritation - need_at_least
            offset = int(len_diff * torch.rand(1))
            images = images[offset:(need_at_least + offset)]
         
        duritation = len(images)
        if duritation < need_at_least:
            images.extend([images[-1]] * (need_at_least - len(images)))
            
        
        duritation = len(images)    
        img_data = torch.stack(images)
        
        return (img_data, duritation, item_class)
        
    def __len__(self):
        return len(self.files_list)
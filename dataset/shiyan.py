import torch
from torch.utils.data import DataLoader, random_split,Dataset

from dataset import TVTDataset

_,_,test_dataset=TVTDataset()
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)  #

for i, (image, label, image_path, label_path) in enumerate(test_dataloader):
    # print(i, image.size(), label.size())
    print(image_path)





import os

parent_folder = "tiny-imagenet-200/train"

wnist_dict = {}

for i, folder in enumerate(os.listdir(parent_folder)): 
    wnist_dict[folder] = f"{i}"

print(wnist_dict['n02509815'])
    
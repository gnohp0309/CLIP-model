import os
import cv2
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from underthesea import word_tokenize
import config


class UITVIC_DATA(Dataset):
    def __init__(self, json_path, img_dir, augment=False, only_image=False):
        self.img_dir = img_dir
        self.augment = augment
        self.only_image = only_image

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)


        self.images = {item['id']: item['file_name'] for item in self.data['images']}
        self.annotations = pd.DataFrame(self.data['annotations']).drop('id', axis=1)
        self.transform = self.get_transforms(augment)

    def get_transforms(self, augment):
        if augment:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_id = row['image_id']
        caption = row['caption']

        img_filename = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_filename)

        img = cv2.imread(img_path)
        if img is None:
            img = torch.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=torch.uint8).numpy()
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        if self.only_image:
            return img

        text_tokenized = word_tokenize(caption, format="text")
        return img, text_tokenized


class KTVIC_DATA(Dataset):
    def __init__(self, json_path, img_dir, augment=False, only_image=False):
        self.img_dir = img_dir
        self.augment = augment
        self.only_image = only_image

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.dup_images = self.get_dup_img()

        self.images = {item['id']: item['filename'] for item in self.data['images']}
        self.annotations = self.data['annotations']

        self.transform = self.get_transforms(augment)

    def get_transforms(self, augment):
        if augment:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def get_dup_img(self):
        dup_images = {}
        for item in self.data['annotations']:

            dup_images[item['id']] = item['image_id']
        return dup_images

    def __len__(self):
        return len(self.dup_images)

    def __getitem__(self, idx):
        anno_ids = list(self.dup_images.keys())
        anno_id = anno_ids[idx]

        img_id = self.dup_images[anno_id]
        img_filename = self.images[img_id]


        caption = ""
        for anno in self.annotations:
            if anno['id'] == anno_id:
                caption = anno['caption']
                break

        img_path = os.path.join(self.img_dir, img_filename)
        img = cv2.imread(img_path)

        if img is None:
            img = torch.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=torch.uint8).numpy()
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        if self.only_image:
            return img

        text_tokenized = word_tokenize(caption, format="text")
        return img, text_tokenized
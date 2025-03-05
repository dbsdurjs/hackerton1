import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader, Dataset
import random
import os
import shutil

class BinaryClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.classes = ['non_soldier', 'camouflage_soldier']  # 0: 비위장군인, 1: 위장군인

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # 위장 군인 클래스의 인덱스를 1로, 나머지는 0으로 변환
        binary_label = 1 if self.dataset.classes[label] == 'camouflage_soldier' else 0
        image_path = self.dataset.samples[idx][0]

        return image, binary_label, image_path

def create_bagging_datasets(dataset, num_subsets=5, subset_size=None):
    if subset_size is None:
        subset_size = len(dataset)

    subsets = []
    for _ in range(num_subsets):
        indices = torch.randint(len(dataset), size=(subset_size,))
        subset = torch.utils.data.Subset(dataset, indices)
        subsets.append(subset)
    return subsets

def make_dataset(path, method="train"):
    if method=="train":
        data_transform = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif method=="test":
        data_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
        ])    
    #dataset 생성
    root_path = path[0]
    
    # ----------------------------------------------------------------------------------------------------------------
    # 최종 데이터 셋을 만들기 위해 파일 복사 및 붙이기 작업 진행 코드

    non_soldier_dataset = ImageFolder(path[1], transform=data_transform)
    real_soldier_dataset = ImageFolder(path[2], transform=data_transform) #test2 - 실제 데이터 셋 불포함 실험

    copy_forest_dataset = non_soldier_dataset.samples
    copy_r_soldier_dataset = real_soldier_dataset.samples #test2 - 실제 데이터 셋 불포함 실험
    copy_s_soldier_dataset = None

    if method == "train":  # 학습 시 실제 데이터 셋 사용
        synthesis_soldier_dataset = ImageFolder(path[5], transform=data_transform)
        copy_s_soldier_dataset = synthesis_soldier_dataset.samples

        # 이미지 개수 만큼 추출
        copy_forest_dataset = random.sample(non_soldier_dataset.samples, 2000)
        copy_r_soldier_dataset = random.sample(real_soldier_dataset.samples, 700) #test2 - 실제 데이터 셋 불포함 실험

    # 데이터 셋 폴더 생성
    os.makedirs(os.path.join(root_path, 'non_soldier'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'camouflage_soldier'), exist_ok=True)

    non_soldier_has_subdirectory = [item for item in os.listdir(os.path.join(root_path, 'non_soldier'))]
    soldier_has_subdirectory = [item for item in os.listdir(os.path.join(root_path, 'camouflage_soldier'))]

    #non_soldier 데이터 셋 만들기
    if not set(non_soldier_has_subdirectory):
        print(f'non_soldier 데이터 셋을 {method}로 복사')
        # 선택된 이미지들을 새 위치로 복사
        for img_path, label in copy_forest_dataset:
            # 이미지 파일 이름 가져오기
            img_filename = os.path.basename(img_path)

            # 새 위치로 이미지 복사
            shutil.copy2(img_path, os.path.join(path[3], img_filename))

    #camouflage_soldier 데이터 셋 만들기
    if not set(soldier_has_subdirectory):
        print(f'soldier 데이터 셋을 {method}로 복사')
        if copy_s_soldier_dataset:
            for img_path, label in copy_s_soldier_dataset:
                img_filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(path[4], img_filename))
        for img_path, label in copy_r_soldier_dataset: #test2 - 실제 데이터 셋 불포함 실험
            img_filename = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(path[4], img_filename))

    # -----------------------------------------------------------------------------------------------------------------
    # 최종 데이터 셋 생성
    
    dataset = BinaryClassificationDataset(root_path, transform=data_transform)

    if method == "train":
        ratio = 0.8

        train_ratio = int(ratio * len(dataset))
        val_ratio = len(dataset) - train_ratio

        print(f'total: {len(dataset)} | train ratio: {train_ratio} | val_ratio: {val_ratio}')

        train_data, val_data = random_split(dataset, [train_ratio, val_ratio])

        train_subsets = create_bagging_datasets(train_data, num_subsets=5)
        data_loader = [DataLoader(subset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
                         for subset in train_subsets]
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    elif method == "test":
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                                 drop_last=True)
        val_loader = None
    return data_loader, val_loader





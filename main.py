from train_model import training
from test import *
from predict_image import *
from one_img_test import *
import random, os
import numpy as np
import torch
    
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":   
    seed_everything(322)
    
    if torch.cuda.is_available():
        print('cuda is available. working on gpu')
        device = torch.device('cuda')
    else:
        print('cuda is not available. working on gpu')
        device = torch.device('cpu')

    # 데이터 셋 생성 및 모델 훈련, 평가
    # num_epochs = training(device)

    # 테스트
    testing(device)
    

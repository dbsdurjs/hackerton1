import os
from dataset import *
from sampling import *
from convert_rg_blindness import *
from model import *
from pretrained_model import *
from show_graph import show_graphs
import sys

non_soldier_path = "../Landscape Classification/Landscape Classification/Training Data"
real_soldier_path = "../camouflage_soldier_dataset/Training"
synthesis_soldier_path = "../sampling_exp"   # 적록색맹 적용
# synthesis_soldier_path = "./sampling_camouflage_soldier"   # 적록색맹 적용 x(test1)

# train_root_path = "./train_dataset_wo_rg(exp)" # 최종 데이터 셋 저장 경로, 적록색맹 데이터 미포함(test1)
# train_root_path = "../train_dataset_wo_real(exp)" # 최종 데이터 셋 저장 경로, 실제 데이터 미포함(test2)
train_root_path = "../train_dataset_wo_syn(exp)" # 최종 데이터 셋 저장 경로, 생성 데이터 미포함
# train_root_path = "../train_dataset_refine-tuning" # 최종 데이터 셋 저장 경로, 재 생성(dreambooth)

non_soldier_target_path = os.path.join(train_root_path, 'non_soldier')  # 최종 데이터 셋 저장 경로
soldier_target_path = os.path.join(train_root_path, 'camouflage_soldier')    # 최종 데이터 셋 저장 경로

rg_blindness_path = '../sampling_exp/rg_blindness_camouflage_soldier_exp'

def training(device):
    #이미지 sampling - 1800
    if not os.path.isdir("../sampling_exp/camouflage_soldier_exp"):  #change
        # sampling_func()
        pass
    
    if not os.listdir(rg_blindness_path):
        pass
        # move_half_images('../sampling_exp/camouflage_soldier_exp', rg_blindness_path)
        # simulate_rg_color_blindness(rg_blindness_path)

    path = [train_root_path, non_soldier_path, real_soldier_path, non_soldier_target_path, soldier_target_path, synthesis_soldier_path]
    #데이터 셋 생성
    train_loader, val_loader = make_dataset(path, method="train")

    ensemble_train_losses, ensemble_val_losses, ensemble_train_accuracy, ensemble_val_accuracy, num_epochs = train_ensemble(
            train_loader, val_loader, device)

    metrics = [
        ('Train Loss', ensemble_train_losses),
        ('Validation Loss', ensemble_val_losses),
        ('Train Accuracy', ensemble_train_accuracy),
        ('Validation Accuracy', ensemble_val_accuracy)
    ]

    show_graphs(metrics, num_epochs)

    return num_epochs
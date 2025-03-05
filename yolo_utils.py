import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import cv2

# 1. 데이터셋 클래스 (이미지, binary label, 이미지 경로 반환)

# 2. 정규화된 YOLO bbox ([x_center, y_center, width, height])를 절대 좌표([x_min, y_min, x_max, y_max])로 변환하는 함수
def yolo_norm_to_xyxy(box, img_width, img_height):
    x_center, y_center, width, height = box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return [x_min, y_min, x_max, y_max]

# 3. IoU 계산 함수 (두 bbox가 [x_min, y_min, x_max, y_max] 형식)
def compute_iou(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

# 4. gt 파일에서 정규화된 bbox 정보를 읽어오는 함수
def load_gt_box(image_path, labels_dir):
    """
    image_path: 예) 'dataset/camouflage_soldier/imagename.jpg'
    labels_dir: gt 파일들이 저장된 폴더 (예: "labels")
    
    이미지 파일명에 해당하는 'imagename.txt' 파일을 읽어 첫 번째 라인의 bbox 반환
    """
    base = os.path.basename(image_path)           # 예: 'imagename.jpg'
    file_name, _ = os.path.splitext(base)          # 예: 'imagename'
    label_file = os.path.join(labels_dir, file_name + ".txt")
    
    boxes = []
    if not os.path.exists(label_file):
        print(f"GT 파일이 존재하지 않습니다: {label_file}")
        return boxes

    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:  # 빈 줄 건너뛰기
            continue
        parts = line.split()
        # 첫 번째 값은 클래스 번호이므로 제외하고, 나머지 4개 값을 float으로 변환
        bbox = list(map(float, parts[1:]))
        boxes.append(bbox)
        
    return boxes

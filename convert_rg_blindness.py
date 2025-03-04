from PIL import Image
import numpy as np

import os
import shutil
import random

def move_half_images(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    # 소스 폴더의 모든 파일 목록 가져오기
    all_files = os.listdir(source_folder)

    # 이미지 파일만 선택
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # 이미지 파일 목록을 무작위로 섞기
    random.shuffle(image_files)

    # 이미지 파일 절반 선택
    num_files_to_move = len(image_files) // 2
    files_to_move = image_files[:num_files_to_move]

    # 파일 이동
    for file_name in files_to_move:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.move(source_path, destination_path)
        print(f"Moved {file_name} to {destination_folder}")

def simulate_rg_color_blindness(images):
    all_files = os.listdir(images)

    for image_path in all_files:
        image_path = os.path.join(images, image_path)
        image = Image.open(image_path)
        # Convert image to numpy array
        image_np = np.array(image)

        # Extract color channels
        R = image_np[:, :, 0]
        G = image_np[:, :, 1]
        B = image_np[:, :, 2]

        # Simulate R-G color blindness
        transformed_R = 0.299 * R + 0.587 * G + 0.114 * B
        transformed_G = 0.299 * R + 0.587 * G + 0.114 * B
        transformed_B = B
        color_blind_image_np = np.stack([transformed_R, transformed_G, transformed_B], axis=-1)

        # Convert back to PIL image
        color_blind_image = Image.fromarray(np.uint8(color_blind_image_np))
        color_blind_image.save(image_path)
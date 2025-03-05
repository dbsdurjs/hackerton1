import os

labels_folder = '/home/yeogeon/YG_main/diffusion_model/diffusers/examples/dreambooth/bbox_dataset/test/labels'

# 폴더 내의 모든 파일 순회
for filename in os.listdir(labels_folder):
    # 확장자가 .txt 인 파일만 처리
    if filename.endswith(".txt"):
        # 파일 이름에서 '_' 기준으로 분할 후 첫 번째 부분 추출
        base_name = filename.split('.')[0]
        new_filename = base_name + ".txt"
        
        # 기존 경로와 새 경로 생성
        old_path = os.path.join(labels_folder, filename)
        new_path = os.path.join(labels_folder, new_filename)
        
        # 같은 이름이 아니라면 파일 이름 변경
        if old_path != new_path:
            print(f"Renaming '{old_path}' to '{new_path}'")
            os.rename(old_path, new_path)

# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors
# import torch
# from predict_image import predicting
# import cv2
# import random

# def detecting(img_path, conf):

#     detect_model = YOLO("yolov8x.pt")
    
#     result = detect_model.predict(img_path, save=False, classes=0, conf=conf)
#     # 이미지 로드
#     img = cv2.imread(img_path)

#     # Annotator 객체 생성
#     annotator = Annotator(img)

# # 결과에서 bbox 정보 추출 및 그리기
#     for r in result:
#         boxes = r.boxes
#         for box in boxes:
#             b = box.xyxy[0].cpu().numpy().astype(int)  # bbox 좌표
#             c = int(box.cls)  # 클래스
#             conf = float(box.conf)  # 확률값
#             label = f"{detect_model.names[c]} {conf:.2f}"  # 클래스 이름과 확률값
#             annotator.box_label(b, label, color=(0, 0, 255))  # 빨간색 (BGR 형식)

#     # 결과 이미지 저장
#     cv2.imwrite(f'./runs/detect/predict{random.randint(0, 1500)}.jpg', annotator.result())
#     box_prob = result[0].boxes.conf
    
#     if box_prob.numel() == 1:
#         return box_prob
#     elif box_prob.numel() > 1:
#         box_prob = torch.max(box_prob).unsqueeze(0)
#         return box_prob
#     else:
#         box_prob = torch.zeros(1)
#         return box_prob
    
# if torch.cuda.is_available():
#         print('cuda is available. working on gpu')
#         device = torch.device('cuda')
# else:
#         print('cuda is not available. working on gpu')
#         device = torch.device('cpu')
        
# # 특정 이미지 테스트
# # predict_img_path = './test_result/real(f) - error/Forest-Valid (292).jpeg'
# predict_img_path = './test_result/real(t) - improve/image558.jpg'

# od_threshold = 0.6
# cls_threshold = 0.5

# while True:
#     # cls prediction
#     is_soldier, cls_prob = predicting(predict_img_path, device, threshold=cls_threshold)
#     # yolo detection
#     box_prob = detecting(predict_img_path, conf=od_threshold)
    
#     # if not is_soldier:
#     #     cls_prob = 1 - cls_prob
        
#     print(f'is soldier {is_soldier} | cls prob {cls_prob:.2f}| cls threshold {cls_threshold} | od box_prob {box_prob.item()} | od threshold {od_threshold}')

#     if (box_prob is None and is_soldier == False) or (box_prob and is_soldier == True): # 둘 다 True, False
#         break

#     if (not is_soldier and box_prob.item()):  # classification 인식 못함, detection 인식함
#         if cls_threshold > 0.1:
#             cls_threshold = round(cls_threshold - 0.1, 2)
#         elif cls_threshold <= 0.01:
#             break
#         else:
#             cls_threshold = round(cls_threshold - 0.01, 2)

#     if (is_soldier and box_prob.item() == 0): # classification 인식함, detection 인식 못함
#         if od_threshold > 0.1:
#             od_threshold = round(od_threshold - 0.1, 2)
#         elif od_threshold <= 0.01:
#             break
#         else:
#             od_threshold = round(od_threshold - 0.01, 2)

# print(f'classifier : {is_soldier}')
# print(f'classified with {cls_prob:.2f} and detected with {box_prob.item()}')
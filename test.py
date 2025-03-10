from model import *
from convert_rg_blindness import *
from pretrained_model import *
import csv, cv2, torchvision
import pandas as pd
from show_graph import show_graphs
from dataset import make_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from ultralytics import YOLO
from yolo_utils import *
import json

non_soldier_path = "../Landscape Classification/Landscape Classification/Validation Data"
real_soldier_path = "../camouflage_soldier_dataset/Testing"
test_root_path = "../test_dataset"

n_target_path = '../bbox_dataset/test/non_soldier'
s_target_path = '../bbox_dataset/test/camouflage_soldier'

labels_folder = '../bbox_dataset/test/labels'

path = [test_root_path, non_soldier_path, real_soldier_path, n_target_path, s_target_path]
test_loader, _ = make_dataset(path, method="test")

checkpoint_dir = '../loss_and_accuracy'
detect_model = YOLO("../yolov8x.pt")

#모델 로드
def load_ensemble_models(ensemble_model, device):
    models = ensemble_model
    # 모델 순서에 맞춰 불러올 파일명을 지정합니다.
    file_names = ['Resnet_models.pt', 'VGG_models.pt', 'Mobilenet_models.pt', 'Dense_models.pt', 'Convnext_models.pt']
    for model, file_name in zip(models, file_names):
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, file_name)))
        model.to(device)
        model.eval()
    return models


def f1_score_cal(tp, fp, fn):
    # Precision과 Recall 계산
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    # Precision과 Recall이 0이면 F1 Score는 0
    if precision + recall == 0:
        print('error about precision and recall')
        return 0
    
    # F1 Score 계산
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def save_dict_to_txt(file_name, dict_data, threshold):
    with open(file_name, 'a') as file:  # 파일을 쓰기 모드로 열기
        for key, value in threshold.items():
            file.write(f"{key} : {value}, ")
        file.write('\n')
        for key, value in dict_data.items():
            file.write(f"{key}: {value}\n")  # 키: 값 형식으로 기록

def save_list_to_txt(file_name, list_data):
    with open(file_name, 'a') as file:  # 파일을 쓰기 모드로 열기
        for value in list_data:
            file.write(value)  # 키: 값 형식으로 기록
            file.write('\n')
            
def calculate_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    return precision, recall, f1, accuracy

def plot_f1_threshold(tp_list_before, fp_list_before, fn_list_before, tn_list_before, 
                      tp_list_after, fp_list_after, fn_list_after, tn_list_after, thresholds):
    
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['figure.titlesize'] = 25
    
    # before metrics 계산
    precision_before = [calculate_metrics(tp, fp, fn, tn)[0] for tp, fp, fn, tn in zip(tp_list_before, fp_list_before, fn_list_before, tn_list_before)]
    recall_before = [calculate_metrics(tp, fp, fn, tn)[1] for tp, fp, fn, tn in zip(tp_list_before, fp_list_before, fn_list_before, tn_list_before)]
    f1_scores_before = [calculate_metrics(tp, fp, fn, tn)[2] for tp, fp, fn, tn in zip(tp_list_before, fp_list_before, fn_list_before, tn_list_before)]
    accuracy_before = [calculate_metrics(tp, fp, fn, tn)[3] for tp, fp, fn, tn in zip(tp_list_before, fp_list_before, fn_list_before, tn_list_before)]

    # After metrics 계산
    precision_after = [calculate_metrics(tp, fp, fn, tn)[0] for tp, fp, fn, tn in zip(tp_list_after, fp_list_after, fn_list_after, tn_list_after)]
    recall_after = [calculate_metrics(tp, fp, fn, tn)[1] for tp, fp, fn, tn in zip(tp_list_after, fp_list_after, fn_list_after, tn_list_after)]
    f1_scores_after = [calculate_metrics(tp, fp, fn, tn)[2] for tp, fp, fn, tn in zip(tp_list_after, fp_list_after, fn_list_after, tn_list_after)]
    accuracy_after = [calculate_metrics(tp, fp, fn, tn)[3] for tp, fp, fn, tn in zip(tp_list_after, fp_list_after, fn_list_after, tn_list_after)]
    
    # 그래프 그리기
    plt.figure(figsize=(12, 8))
    
    # F1 스코어 그래프
    plt.plot(thresholds, f1_scores_before, marker='o', label='Before cora', color='blue', linewidth=2)
    plt.plot(thresholds, f1_scores_after, marker='o', label='After cora', color='red', linewidth=2)
    
    plt.xlabel('Threshold', fontweight='bold')
    plt.ylabel('F1 Score', fontweight='bold')
    plt.title('F1 Score vs Threshold (Before and After)', fontweight='bold')
    plt.grid(True)
    plt.legend()
    
    # 그래프 저장
    plt.savefig("combined_f1_score.png")
    plt.close()
    
    # # Precision과 Recall 그래프 (같은 그래프에 표시)
    # plt.figure(figsize=(10, 6))
    # plt.plot(thresholds, precision_before, marker='o', label='Precision Before cora', color='blue', linestyle='--')
    # plt.plot(thresholds, recall_before, marker='o', label='Recall Before cora', color='blue', linestyle='-')
    # plt.plot(thresholds, precision_after, marker='o', label='Precision After cora', color='red', linestyle='--')
    # plt.plot(thresholds, recall_after, marker='o', label='Recall After cora', color='red', linestyle='-')
    
    # plt.xlabel('Threshold')
    # plt.ylabel('Score')
    # plt.title('Precision and Recall vs Threshold (Before and After)')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig("precision_recall_comparison.png")
    # plt.close()

    # # Accuracy 그래프
    # plt.figure(figsize=(10, 6))
    # plt.plot(thresholds, accuracy_before, marker='o', label='Accuracy Before cora', color='blue', linestyle='-')
    # plt.plot(thresholds, accuracy_after, marker='o', label='Accuracy After cora', color='red', linestyle='-')
    
    # plt.xlabel('Threshold')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs Threshold (Before and After)')
    # plt.grid(True)
    # plt.legend()
    
    # # Accuracy 그래프 저장
    # plt.savefig("accuracy_comparison.png")
    # plt.close()

def cls_measure(pred, target, cls_m):
    if pred and target:
        cls_m['r_t_cls_t'] += 1
    elif not pred and target:
        cls_m['r_t_cls_f'] += 1
    elif pred and not target:
        cls_m['r_t_cls_t'] += 1
    elif not pred and not target:
        cls_m['r_f_cls_f'] += 1

    return cls_m

#이미지 예측
def predict_image(models, device):
    c_test_loss = 0.0
    cls_targets = []
    cls_predictions = []  # is it soldier? (T/F)  
    probability = []    # soldier probability

    od_predictions = []
    
    r_t_cls_f_od_t = [] #실제 true, cls_false, od_true
    
    criterion = nn.BCEWithLogitsLoss().to(device)

    od_threshold = 0.6
    classification_threshold = 0.5
    
    before_f1_score_result_od  = {'0.5' : 0, '0.55' : 0, '0.6' : 0, '0.65' : 0, '0.7' : 0, '0.75' : 0, '0.8' : 0, '0.85' : 0, '0.9' : 0, '0.95' : 0, '1.0' : 0}
    after_f1_score_result_od  = {'0.5' : 0, '0.55' : 0, '0.6' : 0, '0.65' : 0, '0.7' : 0, '0.75' : 0, '0.8' : 0, '0.85' : 0, '0.9' : 0, '0.95' : 0, '1.0' : 0}
    
    thresholds = []
    before_tp_list = []
    before_fp_list = []
    before_fn_list = []
    before_tn_list = []
    
    after_tp_list = []
    after_fp_list = []
    after_fn_list = []
    after_tn_list = []
            
    improvments = []
    errors = []
    
    while True:
        
        if od_threshold > 0.6 : 
            break
        cora_before = {'r_t_cls_od_t' : 0, 'r_f_cls_od_t' : 0, 'r_t_cls_t_od_f' : 0, 'r_t_cls_f_od_t' : 0, 
                    'r_f_cls_t_od_f' : 0, 'r_f_cls_f_od_t' : 0, 'r_t_cls_od_f' : 0, 'r_f_cls_od_f' : 0}
        cora_after = {'r_t_cls_od_t' : 0, 'r_f_cls_od_t' : 0, 'r_t_cls_t_od_f' : 0, 'r_t_cls_f_od_t' : 0, 
                    'r_f_cls_t_od_f' : 0, 'r_f_cls_f_od_t' : 0, 'r_t_cls_od_f' : 0, 'r_f_cls_od_f' : 0}

        # classification vs classification + object detection
        classification_measurement = {'r_t_cls_t' : 0, 'r_f_cls_t' : 0, 'r_t_cls_f' : 0, 'r_f_cls_f' : 0}  
        
        tp = {'cora_before' : 0, 'cora_after' : 0}
        fp = {'cora_before' : 0, 'cora_after' : 0}
        fn = {'cora_before' : 0, 'cora_after' : 0}
        tn = {'cora_before' : 0, 'cora_after' : 0}

        result_json = []
        for idx, (inputs, labels, img_path) in enumerate(test_loader):
            img_path = img_path[0]
            inputs, labels = inputs.to(device), labels.to(device)

            before_result_json = {
                'before img': img_path.split('/')[-1],
                'classification_prediction': 0,
                'classification_conf': 0,
                'classification_threshold': 0,
                'detection_conf': 0,
                'detection_threshold': 0,
                'detection_iou': 0
            }

            after_result_json = {
                'after img': img_path.split('/')[-1],
                'classification_prediction': 0,
                'classification_conf': 0,
                'classification_threshold': 0,
                'detection_conf': 0,
                'detection_threshold': 0,
                'detection_iou': 0
            }
            improve = 0
            err = 0
            with torch.no_grad():
                targets = labels.float().unsqueeze(1)
                
                norm = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # yolo모델은 normalize 입력을 받지 않음, 따로 normalize 해주기
                ensemble_input = norm(inputs)
                ensemble_output = torch.stack([model(ensemble_input) for model in models], dim=0)   # torch.Size([5, 16, 1])
                ensemble_output = torch.mean(ensemble_output, dim=0)    # torch.Size([16, 1])
                
                loss = criterion(ensemble_output, targets)
                c_test_loss += loss.item()

                prob = torch.sigmoid(ensemble_output)
                predicted = prob > classification_threshold   # 군인인지 아닌지
                
                classification_measurement = cls_measure(predicted, targets.item(), classification_measurement)  # classification vs classification + object detection
                
                probability.extend(prob.cpu().numpy())
                cls_targets.extend(targets.cpu().numpy())
                cls_predictions.extend(predicted.cpu().numpy())

                resize = torchvision.transforms.Resize(size=(640, 640)) # 적정 이미지 사이즈로 변환
                inputs = resize(inputs)
                
                detect_result = detect_model.predict(inputs, classes=0, conf=od_threshold, verbose=False)  #nomalize 안된 이미지 입력
                box_prob = detect_result[0].boxes.conf
                box_coor = detect_result[0].boxes.xyxy

                if box_prob.numel() == 0:   # object detection - False
                    before_result_json['classification_prediction'] = predicted.tolist()
                    before_result_json['classification_conf'] = prob.tolist()
                    before_result_json['classification_threshold'] = classification_threshold

                    before_result_json['detection_iou'] = 0
                    before_result_json['detection_conf'] = box_prob.tolist()
                    before_result_json['detection_threshold'] = od_threshold

                    if predicted:   #Classification - True, Object detection - False
                        if targets.item(): # 실제 true, cls_true, od_false
                            cora_before['r_t_cls_t_od_f'] += 1
                            improve += 1
                        else:   # 실제 false, cls_true, od_false
                            cora_before['r_f_cls_t_od_f'] += 1
                            err += 1
                            
                        # cora 적용
                        box_prob, box_coor, detect_result, threshold = cora(inputs, models, detect_model, case='ct_odf', threshold=od_threshold)

                        if box_prob.numel() >= 1:
                            gt_norm_box = load_gt_box(img_path, labels_folder)
                            if not gt_norm_box:  # 리스트가 비어 있으면
                                print(f"{img_path}: GT 파일이 없으므로 IoU 계산을 건너뜁니다.")
                                continue
                            gt_box = [yolo_norm_to_xyxy(gt_norm_box[0], 640, 640)]
                            max_ious = []
                            for pred in box_coor:
                                ious = [compute_iou(pred, gt) for gt in gt_box]
                                max_iou = max(ious) if ious else 0
                                max_ious.append(max_iou.item())
                                print(f'{img_path}, iou:{max_iou}')

                            after_result_json['classification_prediction'] = predicted.tolist()
                            after_result_json['classification_conf'] = prob.tolist()
                            after_result_json['classification_threshold'] = classification_threshold

                            after_result_json['detection_iou'] = max_ious
                            after_result_json['detection_conf'] = box_prob.item()
                            after_result_json['detection_threshold'] = threshold

                            detect_result[0].save(filename= f'../runs/{img_path.split('/')[-1]}')

                            if targets.item():  # cls_true, od_true, 실제 true(개선된 경우)
                                cora_after['r_t_cls_od_t'] += 1
                                if improve and od_threshold == 0.6:
                                    improvments.append(test_loader.dataset.dataset.imgs[idx][0])
                            else:  # cls_true, od_true, 실제 false
                                cora_after['r_f_cls_od_t'] += 1
                                if err and od_threshold == 0.6:
                                    errors.append(test_loader.dataset.dataset.imgs[idx][0])
                        else:   # cls_true, od_false
                            box_prob = torch.zeros(1)
                            if targets.item():  # cls_true, od_false, 실제 true
                                cora_after['r_t_cls_t_od_f'] += 1
                            else:  # cls_true, od_false, 실제 false
                                cora_after['r_f_cls_t_od_f'] += 1

                    elif not predicted and not targets.item(): # cls_false, od_false, 실제 false
                        cora_before['r_f_cls_od_f'] += 1
                    elif not predicted and targets.item():  # cls_false, od_false, 실제 true
                        cora_before['r_t_cls_od_f'] += 1
                        
                elif box_prob.numel() >= 1: # object detection - True
                    gt_norm_box = load_gt_box(img_path, labels_folder)
                    if not gt_norm_box:  # 리스트가 비어 있으면
                        print(f"{img_path}: GT 파일이 없으므로 IoU 계산을 건너뜁니다.")
                        continue
                    gt_box = [yolo_norm_to_xyxy(gt_norm_box[0], 640, 640)]
                    max_ious = []
                    for pred in box_coor:
                        ious = [compute_iou(pred, gt) for gt in gt_box]
                        max_iou = max(ious) if ious else 0
                        max_ious.append(max_iou.item())
                        print(f'{img_path}, iou:{max_iou}')

                    before_result_json['classification_prediction'] = predicted.tolist()
                    before_result_json['classification_conf'] = prob.tolist()
                    before_result_json['classification_threshold'] = classification_threshold

                    before_result_json['detection_iou'] = max_ious
                    before_result_json['detection_conf'] = box_prob.tolist()
                    before_result_json['detection_threshold'] = od_threshold

                    detect_result[0].save(filename= f'../runs/{img_path.split('/')[-1]}')
                    box_prob = torch.max(box_prob).unsqueeze(0)

                    if not predicted:   #Classification - False, Object detection - True
                        if targets.item():  #Classification - False, Object detection - True, 실제 true
                            cora_before['r_t_cls_f_od_t'] += 1
                            improve += 1
                            # r_t_cls_f_od_t.append(test_loader.dataset.dataset.imgs[idx][0])
                        else:
                            cora_before['r_f_cls_f_od_t'] += 1
                            err += 1
                            
                        # cora 적용
                        is_soldier, soldier_prob, _, threshold = cora(inputs, models, detect_model, case='odt_cf', threshold=classification_threshold)
                        
                        after_result_json['classification_prediction'] = is_soldier.tolist()
                        after_result_json['classification_conf'] = soldier_prob.tolist()
                        after_result_json['classification_threshold'] = threshold

                        after_result_json['detection_iou'] = max_ious
                        after_result_json['detection_conf'] = box_prob.tolist()
                        after_result_json['detection_threshold'] = od_threshold

                        if is_soldier:
                            if targets.item() and improve:  # cls_true, od_true, 실제 true
                                cora_after['r_t_cls_od_t'] += 1
                                if improve and od_threshold == 0.6:
                                    improvments.append(test_loader.dataset.dataset.imgs[idx][0])
                            else:  # cls_true, od_true, 실제 false
                                cora_after['r_f_cls_od_t'] += 1
                                if err and od_threshold == 0.6:
                                    errors.append(test_loader.dataset.dataset.imgs[idx][0])
                        else:   # cls_false, od_true
                            if targets.item():  # cls_false, od_true, 실제 true
                                cora_after['r_t_cls_f_od_t'] += 1
                            else:  # cls_false, od_true, 실제 false
                                cora_after['r_f_cls_f_od_t'] += 1
                            
                    elif predicted and targets.item():  # cls_true, od_true, 실제 true
                        cora_before['r_t_cls_od_t'] += 1
                    elif predicted and not targets.item():  # cls_true, od_true, 실제 false
                        cora_before['r_f_cls_od_t'] += 1
                od_predictions.extend(box_prob.cpu().numpy())
                result_json.append(before_result_json)
                result_json.append(after_result_json)

        with open('results.json', 'w') as f:
            json.dump(result_json, f, indent=4)

        cora_after['r_t_cls_od_f'] = cora_before['r_t_cls_od_f']
        cora_after['r_f_cls_od_t'] += cora_before['r_f_cls_od_t']
        cora_after['r_t_cls_od_t'] += cora_before['r_t_cls_od_t']
        cora_after['r_f_cls_od_f'] = cora_before['r_f_cls_od_f']
        
        tp['cora_before'] = cora_before['r_t_cls_od_t']
        fp['cora_before'] = cora_before['r_f_cls_od_t'] + cora_before['r_f_cls_t_od_f'] + cora_before['r_f_cls_f_od_t']
        fn['cora_before'] = cora_before['r_t_cls_od_f'] + cora_before['r_t_cls_t_od_f'] + cora_before['r_t_cls_f_od_t']
        tn['cora_before'] = cora_before['r_f_cls_od_f']
        
        tp['cora_after'] = cora_after['r_t_cls_od_t']
        fp['cora_after'] = cora_after['r_f_cls_od_t'] + cora_after['r_f_cls_t_od_f'] + cora_after['r_f_cls_f_od_t']
        fn['cora_after'] = cora_after['r_t_cls_od_f'] + cora_after['r_t_cls_t_od_f'] + cora_after['r_t_cls_f_od_t']
        tn['cora_after'] = cora_after['r_f_cls_od_f']
        
        save_dict_to_txt('cora_before', cora_before, threshold={'cls_threshold' : classification_threshold, 'od_threshold' : od_threshold})
        save_dict_to_txt('cora_after', cora_after, threshold={'cls_threshold' : classification_threshold, 'od_threshold' : od_threshold})
        # save_dict_to_txt('classifcation_measurement', classification_measurement, threshold={'cls_threshold' : classification_threshold, 'od_threshold' : od_threshold})
        
        before_tp_list.append(tp['cora_before'])
        before_fp_list.append(fp['cora_before'])
        before_fn_list.append(fn['cora_before'])
        before_tn_list.append(tn['cora_before'])
        
        after_tp_list.append(tp['cora_after'])
        after_fp_list.append(fp['cora_after'])
        after_fn_list.append(fn['cora_after'])
        after_tn_list.append(tn['cora_after'])
        
        _, _, cora_before_f1_score, _ = calculate_metrics(tp['cora_before'], fp['cora_before'], fn['cora_before'], tn['cora_before'])
        _, _, cora_after_f1_score, _ = calculate_metrics(tp['cora_after'], fp['cora_after'], fn['cora_after'], tn['cora_after'])
        _, _, cls_f1_score, _ = calculate_metrics(classification_measurement["r_t_cls_t"], classification_measurement["r_f_cls_t"], classification_measurement["r_t_cls_f"], classification_measurement["r_f_cls_f"])

        for key in before_f1_score_result_od.keys():
            if str(od_threshold) == key:
                before_f1_score_result_od[key] = cora_before_f1_score
                thresholds.append(key)
        
        for key in after_f1_score_result_od.keys():
            if str(od_threshold) == key:
                after_f1_score_result_od[key] = cora_after_f1_score
                
        od_threshold = round(od_threshold + 0.05, 2)

    save_list_to_txt('improve list',improvments)
    save_list_to_txt('error list',errors)
    
    plot_f1_threshold(before_tp_list, before_fp_list, before_fn_list, before_tn_list, 
                      after_tp_list, after_fp_list, after_fn_list, after_tn_list, thresholds)
    print(f'before cora f1 score {before_f1_score_result_od}')
    print(f'after cora f1 score {after_f1_score_result_od}')
    print(f'cls f1 score {cls_f1_score}')

    # Classification 정확도
    test_accuracy = 100 * (torch.tensor(cls_predictions) == torch.tensor(cls_targets)).float().mean()
    c_test_loss /= len(test_loader)
    
    print(f'classification Test Accuracy: {test_accuracy:.2f}%')
    print(f'classification Test Loss: {c_test_loss:.4f}')
    
    print(f'before tp list : {before_tp_list}')
    print(f'before tn list : {before_tn_list}')
    print(f'before fp list : {before_fp_list}')
    print(f'before fn list : {before_fn_list}')
    
    print(f'after tp list : {after_tp_list}')
    print(f'after tn list : {after_tn_list}')
    print(f'after fp list : {after_fp_list}')
    print(f'after fn list : {after_fn_list}')
    
    print(classification_measurement)
    print(f'cls tp list : {classification_measurement["r_t_cls_t"]}')
    print(f'cls tn list : {classification_measurement["r_f_cls_f"]}')
    print(f'cls fp list : {classification_measurement["r_f_cls_t"]}')
    print(f'cls fn list : {classification_measurement["r_t_cls_f"]}')

def testing(device):
    # 모델 로드 및 예측
    loaded_models = load_ensemble_models(generate_model(), device)
    
    predict_image(loaded_models, device)
    
def cora(inputs, models, detect_model, case, threshold):

    # cora 알고리즘
    while True:
        if case == 'odt_cf':  # classification False, object detection True
            if threshold > 0.1:
                threshold = round(threshold - 0.1, 2)
            elif threshold <= 0.01:
                return predicted, prob, _, threshold
            else:
                threshold = round(threshold - 0.01, 2)
            
            with torch.no_grad():
                norm = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # yolo모델은 normalize 입력을 받지 않음, 따로 normalize 해주기
                ensemble_input = norm(inputs)
                
                ensemble_output = torch.stack([model(ensemble_input) for model in models], dim=0)
                ensemble_output = torch.mean(ensemble_output, dim=0)
                prob = torch.sigmoid(ensemble_output)
                predicted = prob > threshold
                
                if predicted.item():
                    return predicted, prob, _, threshold

        elif case == 'ct_odf':    # classification True, object detection False
            if threshold > 0.1:
                threshold = round(threshold - 0.1, 2)
            elif threshold <= 0.01:
                return box_prob, box_coor, detect_result, threshold
            else:
                threshold = round(threshold - 0.01, 2)

            detect_result = detect_model.predict(inputs, classes=0, conf=threshold, verbose=False)  #nomalize 안된 이미지 입력
            box_prob = detect_result[0].boxes.conf
            box_coor = detect_result[0].boxes.xyxy

            if box_coor.numel == 0:
                box_coor = None

            if box_prob.numel() == 1: # object detection 1명 탐지
                box_prob = box_prob
                return box_prob, box_coor, detect_result, threshold
            
            elif box_prob.numel() >= 1:   # object detection 여러 명 탐지(max 값 적용)
                box_prob = torch.max(box_prob).unsqueeze(0)
                return box_prob, box_coor, detect_result, threshold
        
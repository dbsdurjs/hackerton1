import torch
import torch.nn as nn
import torchvision.models as models

# 가중치 초기화 함수 (nn.Conv2d와 nn.Linear에 대해 적용)
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# 기본 모델 생성 함수
def get_pretrained_model(model_name, num_classes=1):
    if isinstance(model_name, models.ResNet):
        model_name.fc = nn.Sequential(
            nn.Linear(model_name.fc.in_features, num_classes),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, num_classes),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(128, 64),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(64, num_classes)
        )
        # fc 부분에만 가중치 초기화 적용
        model_name.fc.apply(init_weights)

    elif isinstance(model_name, models.VGG):
        model_name.classifier[6] = nn.Sequential(
            nn.Linear(model_name.classifier[6].in_features, num_classes),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, num_classes),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(128, 64),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(64, num_classes)
        )
        # classifier의 해당 부분에만 가중치 초기화 적용
        model_name.classifier[6].apply(init_weights)

    elif isinstance(model_name, models.MobileNetV2):
        model_name.classifier[1] = nn.Sequential(
            nn.Linear(model_name.classifier[1].in_features, num_classes),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, num_classes),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(128, 64),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(64, num_classes)
        )
        # classifier의 해당 부분에만 가중치 초기화 적용
        model_name.classifier[1].apply(init_weights)

    elif isinstance(model_name, models.DenseNet):
        model_name.classifier = nn.Sequential(
            nn.Linear(model_name.classifier.in_features, num_classes),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, num_classes),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(128, 64),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(64, num_classes)
        )
        # classifier 전체에 가중치 초기화 적용
        model_name.classifier.apply(init_weights)

    elif isinstance(model_name, models.ConvNeXt):
        model_name.classifier[2] = nn.Sequential(
            nn.Linear(model_name.classifier[2].in_features, num_classes),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, num_classes),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(128, 64),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(64, num_classes)
        )
        # classifier의 해당 부분에만 가중치 초기화 적용
        model_name.classifier[2].apply(init_weights)

    # L2 정규화 (모든 파라미터에 대해 hook 등록)
    for param in model_name.parameters():
        param.register_hook(lambda grad: grad + 0.01 * param)

    return model_name


def generate_model():
    # 사전 학습된 모델 불러오기
    resnet50 = models.resnet50(weights=None)
    vgg16 = models.vgg16(weights=None)
    mobilenet_v2 = models.mobilenet_v2(weights=None)
    densenet121 = models.densenet121(weights=None)
    convnext_tiny = models.convnext_tiny(weights=None)

    ensemble_model_list = [resnet50, densenet121, vgg16, mobilenet_v2, convnext_tiny]
    ensemble_model = [get_pretrained_model(model) for model in ensemble_model_list]

    return ensemble_model

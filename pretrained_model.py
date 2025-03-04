import torch
import torch.nn as nn
import torchvision.models as models

# 기본 모델 생성 클래스
def get_pretrained_model(model_name, num_classes=1):
    if isinstance(model_name, models.ResNet):
        model_name.fc = nn.Sequential(
            nn.Linear(model_name.fc.in_features, 512), #512->256
            nn.LeakyReLU(), #relu -> leakyrelu
            nn.Dropout(0.5),    #0.5->0.7
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)

        )
    elif isinstance(model_name, models.VGG):
        model_name.classifier[6] = nn.Sequential(
            nn.Linear(model_name.classifier[6].in_features, 512),
            nn.LeakyReLU(), #relu -> leakyrelu
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    elif isinstance(model_name, models.MobileNetV2):
        model_name.classifier[1] = nn.Sequential(
            nn.Linear(model_name.classifier[1].in_features, 512),
            nn.LeakyReLU(), #relu -> leakyrelu
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    elif isinstance(model_name, models.DenseNet):
        model_name.classifier = nn.Sequential(
            nn.Linear(model_name.classifier.in_features, 512),
            nn.LeakyReLU(), #relu -> leakyrelu
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    elif isinstance(model_name, models.ConvNeXt):
        model_name.classifier[2] = nn.Sequential(
            nn.Linear(model_name.classifier[2].in_features, 512),
            nn.LeakyReLU(),  #relu -> leakyrelu
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    # L2 regularization 적용
    for param in model_name.parameters():
        param.register_hook(lambda grad: grad + 0.01 * param)

    return model_name


def generate_model():
    # 모델 생성
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    densenet121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    convnext_tiny = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

    ensemble_model_list = [resnet50, vgg16, mobilenet_v2, densenet121, convnext_tiny]
    ensemble_model = [get_pretrained_model(model) for model in ensemble_model_list]

    return ensemble_model
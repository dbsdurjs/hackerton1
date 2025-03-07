import torch
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self, num_classes=1):
        super(Model1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 240, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=3, stride=3),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(15360, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Model2(nn.Module):
    def __init__(self, num_classes=1):
        super(Model2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 240, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=5, stride=3),  # padding=0 (기본값)
            nn.AvgPool2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=3, stride=3),  # output: 2×2
            nn.AvgPool2d(kernel_size=2, stride=2),          # output: 1×1
            nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1),  # 유지 1×1
            nn.AvgPool2d(kernel_size=1, stride=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(240, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Model3(nn.Module):
    def __init__(self, num_classes=1):
        super(Model3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 240, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=7, padding=2),
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(240, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Model4(nn.Module):
    def __init__(self, num_classes=1):
        super(Model4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 240, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=3, stride=3),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=3, stride=3),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 변경: 3→2
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(240, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Model5(nn.Module):
    def __init__(self, num_classes=1):
        super(Model5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 240, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=3, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=3, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(2160, 1024),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    
def generate_model():
    # 모델 생성
    modle1 = Model1()
    model2 = Model2()
    model3 = Model3()
    model4 = Model4()
    model5 = Model5()

    ensemble_model_list = [modle1, model2, model3, model4, model5]

    return ensemble_model_list
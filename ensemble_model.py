import torch
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self, num_classes=10):
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
            nn.Linear(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Model2(nn.Module):
    def __init__(self, num_classes=10):
        super(Model2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 240, kernel_size=3, padding=1),
            nn.AveragePooling2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=5, stride=3),
            nn.AveragePooling2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=3, stride=3),
            nn.AveragePooling2d(kernel_size=3, stride=3),
            nn.Conv2d(240, 240, kernel_size=3, stride=3),
            nn.AveragePooling2d(kernel_size=3, stride=3),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Model3(nn.Module):
    def __init__(self, num_classes=10):
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
            nn.Linear(),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Model4(nn.Module):
    def __init__(self, num_classes=10):
        super(Model4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 240, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=3, stride=3),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(240, 240, kernel_size=3, stride=3),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Model5(nn.Module):
    def __init__(self, num_classes=10):
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
            nn.Linear(),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# as they are not standard PyTorch layers
class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        # Placeholder for custom Linear transformation
        # You may need to implement this based on your specific requirements
        pass
    
    def forward(self, x):
        # Placeholder implementation
        return x
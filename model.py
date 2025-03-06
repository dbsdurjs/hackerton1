import torch.nn.functional
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from early_stopping import *
from pretrained_model import *

def train_ensemble(train_loader, val_loader, device):
    # 모든 모델 인스턴스화
    models = generate_model()
    models = models.to(device)

    # 손실을 저장할 리스트
    ensemble_train_losses = []
    ensemble_train_accuracy = []
    ensemble_val_losses = []
    ensemble_val_accuracy = []

    # 각 모델에 대한 옵티마이저 생성
    optimizers = optim.Adam(models.parameters(), lr=1e-4)
    schedulers = ReduceLROnPlateau(optimizers, 'min', patience=5, factor=0.5)

    # 손실 함수
    criterion = nn.BCEWithLogitsLoss().to(device)
    es = EarlyStopping(patience=5, verbose=True, delta=0.001, path='../loss_and_accuracy')

    # 학습 루프
    num_epochs = 100
    print('use bagging')

    for epoch in range(num_epochs):
        models.train()

        train_loss = 0.0
        train_total = 0
        train_correct = 0

        for i, bag_loader in enumerate(train_loader):
            bag_loss = 0.0
            bag_total = 0
            bag_correct = 0
            for inputs, labels, _ in bag_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                targets = labels.float().unsqueeze(1)

                # i번째 모델로 예측
                optimizers.zero_grad()
                outputs = models(inputs)
                loss = criterion(outputs, targets)
                bag_loss += loss.item()

                # 정확도 계산
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                bag_total += len(targets)    #targets.size(0)
                bag_correct += (predicted == targets).float().sum().item()  #bag_correct += (predicted == targets).sum().item()

                # 역전파 및 최적화
                loss.backward()
                optimizers.step()

            # 각 bag의 loss와 정확도를 전체에 더함
            train_loss += bag_loss
            train_total += bag_total
            train_correct += bag_correct

            train_accuracy = 100 * train_correct / train_total
            train_loss /= len(bag_loader)  # 평균 손실 계산

        # for scheduler in schedulers:
        #     scheduler.step(train_loss)

        ensemble_train_losses.append(train_loss)
        ensemble_train_accuracy.append(train_accuracy)

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            models.eval()

            for inputs, labels, _ in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                targets = labels.float().unsqueeze(1)

                outputs = models(inputs)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += len(targets)   #targets.size(0)
                val_correct += (predicted == targets).float().sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct/val_total

        ensemble_val_losses.append(val_loss)
        ensemble_val_accuracy.append(val_accuracy)

        # 학습률 조정
        schedulers.step(val_loss)

        es.__call__(val_loss, models)
        if es.early_stop:
            num_epochs = epoch
            break

        print(f"epoch [{epoch+1}/{num_epochs}]")

        print(f"train loss: {train_loss:.4f}")
        print(f"train accuracy: {train_accuracy:.2f}")

        print(f"validation loss: {val_loss:.4f}")
        print(f"validation accuracy: {val_accuracy:.2f}")
        print('-'*50)

    return ensemble_train_losses, ensemble_val_losses, ensemble_train_accuracy, ensemble_val_accuracy, num_epochs
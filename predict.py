import torch
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torch import nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, inner_channels, stride=1, projection=None):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels * self.expansion, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels * self.expansion),
        )
        self.projection = projection
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        if self.projection is not None:
            skip_connection = self.projection(x)
        else:
            skip_connection = x
        out = self.relu(residual + skip_connection)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, inner_channels, stride=1, projection=None):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels * self.expansion, 1, bias=False),
            nn.BatchNorm2d(inner_channels * self.expansion),
        )
        self.projection = projection
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        if self.projection is not None:
            skip_connection = self.projection(x)
        else:
            skip_connection = x
        out = self.relu(residual + skip_connection)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_block_list, n_classes=1000):
        super().__init__()
        assert len(num_block_list) == 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.in_channels = 64
        self.stage1 = self.make_stage(block, 64, num_block_list[0], stride=1)
        self.stage2 = self.make_stage(block, 128, num_block_list[1], stride=2)
        self.stage3 = self.make_stage(block, 256, num_block_list[2], stride=2)
        self.stage4 = self.make_stage(block, 512, num_block_list[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, n_classes)

    def make_stage(self, block, inner_channels, num_blocks, stride=1):
        if stride != 1 or self.in_channels != inner_channels * block.expansion:
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, inner_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_channels * block.expansion),
            )
        else:
            projection = None

        layers = []
        for idx in range(num_blocks):
            if idx == 0:
                layers.append(block(self.in_channels, inner_channels, stride, projection))
                self.in_channels = inner_channels * block.expansion
            else:
                layers.append(block(self.in_channels, inner_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# 모델 로드
model = resnet50(n_classes=2)
model.load_state_dict(torch.load('resnet50_model.pth'))  # 저장된 가중치 로드
model.eval()  # 평가 모드 설정

# 이미지 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 데이터셋 로드 및 클래스 목록 확인
train_dataset = ImageFolder(root='dataset3/train')
print(f"클래스 목록: {train_dataset.classes}")


# 예측 함수 정의
def predict_image(image_path):
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        class_name = train_dataset.classes[predicted_class.item()]

    return predicted_class.item(), class_name


# 이미지 예측 실행
image_path = 'test_image/WIN_20241223_16_19_06_Pro.jpg'  # 예측할 이미지 경로
predicted_label, class_name = predict_image(image_path)
print(f"예측 결과: 클래스 {predicted_label}, 클래스 이름: {class_name}")

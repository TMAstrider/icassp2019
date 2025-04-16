import torch
import torch.nn as nn
import torch.nn.functional as F
# import dataloader

class BaselineCNN(nn.Module):
    def __init__(self, n_mels, patch_len, n_classes):
        super(BaselineCNN, self).__init__()
        self.n_mels = n_mels
        self.patch_len = patch_len
        self.n_classes = n_classes

        # 卷积块1
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2))
        )

        # 卷积块2
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2))
        )

        # 卷积块3
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        # 计算展平后的尺寸
        flattened_size = 48 * (patch_len // 16) * (n_mels // 4)
        if flattened_size == 0:
            raise ValueError("Invalid patch_len or n_mels for pooling layers")

        # 全连接部分
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x  # 输出logits，训练时配合CrossEntropyLoss使用


# 使用示例
if __name__ == '__main__':
    # 参数设置
    params_extract = {'patch_len': 100, 'n_mels': 96}  # 示例值
    params_learn = {'n_classes': 20}  # 示例值


    # 创建模型
    model = BaselineCNN(
        n_mels=params_extract['n_mels'],
        patch_len=params_extract['patch_len'],
        n_classes=params_learn['n_classes']
    )

    # 测试输入
    dummy_input = torch.randn(4, 1, params_extract['patch_len'], params_extract['n_mels'])
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

import yaml
import pandas as pd
import numpy as np


import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from baseline_cnn import BaselineCNN

from data import DataGeneratorPatch
from feat_ext import load_audio_file, modify_file_variable_length, get_mel_spectrogram
import utils





print('Import success! \nReady to go!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}\n")

with open('config/params.yaml', 'r') as file:
    config = yaml.safe_load(file)
params_extract = config['extract']
params_learn = config['learn']
params_paths = config['paths']

# params_paths
# params_extract



# 1. 加载 Train csv 数据
train_csv = pd.read_csv(params_paths['train_csv'])
train_csv.head()


# 2. 分离干净/噪声数据
clean_idx = train_csv[train_csv['manually_verified'] == 1].index  # 干净数据索引
noisy_idx = train_csv[train_csv['manually_verified'] == 0].index  # 噪声数据索引

# 3. 提取噪声样本ID (从文件名提取数字部分)
noisy_ids = [int(fname.split('.')[0]) for fname in train_csv.loc[noisy_idx, 'fname']]

# 4. 创建标签映射
labels = sorted(train_csv['label'].unique())    # 所有标签（按字母顺序）（唯一）
label_to_int = {label: i for i, label in enumerate(labels)} # 标签到数字的映射
int_to_label = {i: label for label, i in label_to_int.items()}

# 5. 创建文件路径到标签的映射
file_to_label = {
    f"{params_paths['train_audio_input']}/{row.fname}": row.label 
    for _, row in train_csv.iterrows()
}

# 6. 创建文件路径到数字标签的映射
file_to_int = {
    path: label_to_int[label] 
    for path, label in file_to_label.items()
}

# 打印检查
print(f"干净数据数量: {len(clean_idx)}")
print(f"噪声数据数量: {len(noisy_idx)}\n")
print(f"标签映射示例: {label_to_int}\n")
print(f"前5个文件标签: {list(file_to_label.items())[:5]}\n")
# file_to_int
# file_to_label
# label_to_int


# 1. 加载 test csv 数据
test_csv = pd.read_csv(params_paths['test_csv'])
test_csv.head()

# 4. 创建标签映射
test_labels = sorted(test_csv['label'].unique())    # 所有标签（按字母顺序）（唯一）
test_label_to_int = {label: i for i, label in enumerate(test_labels)} # 标签到数字的映射
test_int_to_label = {i: label for label, i in test_label_to_int.items()}

# 5. 创建文件路径到标签的映射
test_file_to_label = {
    f"{params_paths['test_audio_input']}/{row.fname}": row.label 
    for _, row in test_csv.iterrows()
}

# 6. 创建文件路径到数字标签的映射
test_file_to_int = {
    path: test_label_to_int[label] 
    for path, label in test_file_to_label.items()
}
print(f"标签映射示例: {test_label_to_int}\n")
print(f"前5个文件标签: {list(test_file_to_label.items())[:5]}\n")

# test_file_to_int


# Mel 频谱图的生成




def extract_features(input_dir, output_dir, force_reprocess=False):
    """
    使用原作者的工具函数提取特征
    """
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    # print(audio_files)
    
    # 检查 .data 结尾的文件，存在则替换成 .wav，检查其他还未处理的数据
    if not force_reprocess:
        existing_features = {f.replace('_mel.data', '.wav') for f in os.listdir(output_dir) if f.endswith('_mel.data')}
        files_to_process = [f for f in audio_files if f not in existing_features]
    else:
        files_to_process = audio_files

    if not files_to_process:
        print("\n所有特征文件已存在，无需处理\n")
        return
    
    pbar = tqdm(files_to_process, desc="Extracting features")
    
    for fname in pbar:
        try:
            audio_path = os.path.join(input_dir, fname)
            # print(audio_path)
            # 使用原作者的音频加载函数
            y = load_audio_file(audio_path, 
                              input_fixed_length=params_extract['audio_len_s'],
                              params_extract=params_extract)
            # print(audio_path)
            # print(f"Loaded audio shape: {y.shape}")  # 打印加载的音频形状

            # 使用原作者的长度调整函数
            y = modify_file_variable_length(y,
                                         input_fixed_length=params_extract['audio_len_s'],
                                         params_extract=params_extract)
            # print(audio_path)
            # 使用原作者的梅尔频谱计算函数
            mel_spec = get_mel_spectrogram(y, params_extract)
            # print(audio_path)
            # print(f"Mel spectrogram shape: {mel_spec.shape}")  # 打印梅尔频谱图的形状
            # print()

            output_path = os.path.join(output_dir, fname.replace('.wav', '.data'))
            utils.save_tensor(var=mel_spec, 
                            out_path=output_path, 
                            suffix='_mel')
            # print(audio_path)


            # 保存标签 - 使用file_to_int获取正确的标签索引
            if 'test' in audio_path:
                # print(audio_path, 'test in audio')
                label_idx = test_file_to_int[audio_path]
            else:
                label_idx = file_to_int[audio_path]  # 从映射字典获取标签索引
                
            # print(audio_path)
            utils.save_tensor(var=np.array([label_idx], dtype=float),
                            out_path=output_path,
                            suffix='_label')
            # print(audio_path)
            pbar.set_postfix({'status': f'Processed {fname}'})
            
        except Exception as e:
            print(f"\nError processing {fname}: {str(e)}")
            continue

# 输入输出路径配置
input_dirs = [
    (params_paths['test_audio_input'], params_paths['test_feature_extracted']),  # (输入目录, 输出目录)
    (params_paths['train_audio_input'], params_paths['train_feature_extracted']),  # (输入目录, 输出目录)
]

# 处理所有输入目录
for input_dir, output_dir in input_dirs:
    print(f"\nStarting feature extraction from {input_dir} to {output_dir}")
    extract_features(input_dir, output_dir, force_reprocess=False)
    print(f"Feature extraction from {input_dir} completed!")

print("\nAll feature extraction tasks finished!")



# mel_spec_test_path = "./features/audio_test_varup2"     # (输入目录, 输出目录)
# mel_spec_train_path = "./features/audio_train_varup2"    # (输入目录, 输出目录)

# dataset = 'FSDnoisy18k'
# train_dataset_path = ./fsd18kdataset/FSDnoisy18k.audio_train
# test_dataset_path =./fsd18kdataset/FSDnoisy18k.audio_test
# train_csv_path =./fsd18kdataset/FSDnoisy18k.meta/train.csv
# test_csv_path =./fsd18kdataset/FSDnoisy18k.meta/test.csv




# 设置特征目录
feature_dir = params_paths['test_feature_extracted']

# 获取前5个特征文件
data_files = [f for f in os.listdir(feature_dir) if f.endswith('_mel.data')][:5]

# 读取并打印特征和标签
for data_file in data_files:
    # 构建完整路径
    base_path = os.path.join(feature_dir, data_file.replace('_mel.data', ''))
    
    # 读取特征
    features = utils.load_tensor(base_path + '.data', suffix='_mel')
    
    # 读取标签
    labels = utils.load_tensor(base_path + '.data', suffix='_label')
    
    print(f"\n文件: {data_file}")
    print(f"特征形状: {features.shape}, 数据类型: {features.dtype}")
    print(f"标签值: {labels}")
    print("特征数据片段:")
    print(features[:2, :5])  # 打印前2帧的前5个特征值





class TorchDataWrapper(Dataset):
    def __init__(self, keras_data_gen):
        self.keras_gen = keras_data_gen
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __len__(self):
        return len(self.keras_gen)
    
    def __getitem__(self, idx):
        features, labels = self.keras_gen[idx]
        return (
            # torch.from_numpy(features).float().to(self.device),
            # torch.from_numpy(labels.argmax(1)).long().to(self.device),
            torch.from_numpy(features).float(),
            torch.from_numpy(labels.argmax(1)).long()
        )




# 初始化数据生成器
feature_dir = params_paths['train_feature_extracted']
file_list = [f for f in os.listdir(feature_dir) if f.endswith('_mel.data')]


data_gen = DataGeneratorPatch(
    feature_dir=feature_dir,
    file_list=file_list,
    params_learn=params_learn,
    params_extract=params_extract,
    suffix_in='_mel',
    suffix_out='_label'
)

# 创建PyTorch兼容的数据集
torch_dataset = TorchDataWrapper(data_gen)

# 创建PyTorch DataLoader
train_loader = DataLoader(
    torch_dataset,
    batch_size=None,  # 因为原作者已处理批次
    shuffle=True,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

# 获取并打印第一个batch
for batch_idx, (features, labels) in enumerate(train_loader):
    features = features.to(device)
    labels = labels.to(device)
    
    print(f"\nPyTorch DataLoader 第一个batch:")
    print(f"特征张量形状: {features.shape}")  # 应该是 [batch, 1, time, freq]
    print(f"标签张量形状: {labels.shape}")    # 应该是 [batch]
    
    # 打印第一个样本的部分数据
    print("\n第一个样本的特征数据(部分):")
    print(features[0, 0, :5, :5])  # 打印第一个样本的5x5片段
    
    print("\n所有样本的标签:")
    print(labels)
    
    break  # 只查看第一个batch




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("\nBaselineCNN imported successfully!")
model = BaselineCNN(
    n_mels=params_extract['n_mels'],
    patch_len=params_extract['patch_len'],
    n_classes=params_learn['n_classes']
).to(device)

# 添加模型权重加载功能
pretrained_path = params_paths['pretrained']  # 替换为你的预训练模型路径
if os.path.exists(pretrained_path):
    model.load_state_dict(torch.load(pretrained_path))
    print('\n成功加载预训练权重')
else:
    print('\n未找到预训练模型，将从零开始训练')


print('\nloading optimizer and loss function...')
# 定义优化器和损失函数
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-2,
    weight_decay=1e-3  # L2正则化
)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',  # 监控验证准确率
    factor=0.5,
    patience=5,
    verbose=True
)


# 添加检查点恢复功能
checkpoint_path = 'interrupted_checkpoint.pth'
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    print(f'\n从检查点恢复训练，epoch={start_epoch}, 最佳准确率={best_acc:.4f}')


# 创建TensorBoard writer
writer = SummaryWriter()

# 训练循环
def train(model, loader, optimizer, criterion, epochs=160):
    model.train()
    best_acc = 0.0

    # 添加检查点目录
    os.makedirs('checkpoints', exist_ok=True)

    # 添加epoch进度条
    epoch_pbar = tqdm(range(epochs), desc='Training', unit='epoch')

    for epoch in epoch_pbar:
        try:
            total_loss = 0
            correct = 0
            # 添加batch进度条
            batch_pbar = tqdm(loader, desc=f'Epoch {epoch+1}', leave=False)
            for features, labels in batch_pbar:
                features = features.to(device)
                labels = labels.to(device)
    
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(features)
                loss = criterion(outputs, labels.squeeze())
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.squeeze()).sum().item()

                # 更新batch进度条
                batch_pbar.set_postfix(loss=loss.item())
            
            # 计算epoch统计
            avg_loss = total_loss / len(loader)
            accuracy = correct / len(loader.dataset)

             # 记录到TensorBoard
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Accuracy/train', accuracy, epoch)

            # 更新学习率
            scheduler.step(accuracy)

            # 保存最佳模型
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'保存最佳模型，准确率: {accuracy:.4f}')
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}')

            # 更新epoch进度条
            epoch_pbar.set_postfix(loss=avg_loss, acc=accuracy)

            # # 保存检查点(每个epoch都保存)
            # checkpoint_path = f'checkpoints/epoch_{epoch+1}.pth'
            # torch.save({
            #     'epoch': epoch+1,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'best_acc': best_acc,
            #     'loss': avg_loss,
            # }, checkpoint_path)
        except KeyboardInterrupt:
            print("\n训练被中断，正在保存当前状态...")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'loss': avg_loss,
            }, 'interrupted_checkpoint.pth')
            print("已保存中断检查点到 interrupted_checkpoint.pth")
            return

    # 训练结束后保存最终模型
    # torch.save(model.state_dict(), 'final_model.pth')
    # print('训练完成，最终模型已保存')
    writer.close()
    torch.save(model.state_dict(), 'final_model.pth')
    epoch_pbar.write('训练完成，最终模型已保存')

print('training started...')

# 开始训练
train(model, train_loader, optimizer, criterion, params_learn['n_epochs'])



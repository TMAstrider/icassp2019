# %%
import yaml
import pandas as pd
import numpy as np
import torch
print('Import success! \nReady to go!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('config/params_server.yaml', 'r') as file:
    config = yaml.safe_load(file)
params_extract = config['extract']
params_learn = config['learn']
params_paths = config['paths']
params_ctrl = config['ctrl']
params_suffix = config['suffix']

suffix_in = params_suffix['in']
suffix_out = params_suffix['out']

params_paths
# params_extract


# %%
# 1.1 加载 Train csv 数据
train_csv = pd.read_csv(params_paths['train_csv'])
train_csv.head()

# 1.2 加载 test csv 数据
test_csv = pd.read_csv(params_paths['test_csv'])
test_csv.head()

# 4. 创建标签映射
labels = sorted(train_csv['label'].unique())    # 所有标签（按字母顺序）（唯一）
label_to_int = {label: i for i, label in enumerate(labels)} # 标签到数字的映射
int_to_label = {i: label for label, i in label_to_int.items()}

# 5. 创建文件路径到标签的映射
file_to_label = {
    f"{params_paths['train_audio_input']}/{row.fname}": row.label 
    for _, row in train_csv.iterrows()
}

# file_to_label = {
#     f"{params_paths['train_audio_input']}/{row.fname}": row.label 
#     for _, row in train_csv.iterrows()
# }

# 6. 创建文件路径到数字标签的映射
file_to_int = {
    path: label_to_int[label] 
    for path, label in file_to_label.items()
}

# 打印检查

print(f"标签映射示例: {label_to_int}")
print(f"前5个文件标签: {list(file_to_label.items())[:5]}")
# print(f"查找一个文件的类别：{file_to_int['../fsd18kdataset/FSDnoisy18k.audio_train/94322.wav']}")
# 查找fname等于特定值的记录
result = train_csv[train_csv['fname'] == '171479.wav']

print(result)
file_to_int
file_to_label
label_to_int

###############

filelist_audio_train = train_csv.fname.values.tolist()
filelist_audio_test = test_csv.fname.values.tolist()

# 2.1 提取人类验证过的标注数据
# get positions of manually_verified clips: separate between CLEAN and NOISY sets
train_file_verified_list_col = train_csv.manually_verified.values.tolist()
clean_set = [i for i, x in enumerate(train_file_verified_list_col) if x == 1]
noisy_set = [i for i, x in enumerate(train_file_verified_list_col) if x == 0]

# 2.2 分离干净/噪声数据（行索引
clean_idx = train_csv[train_csv['manually_verified'] == 1].index  # 干净数据索引
noisy_idx = train_csv[train_csv['manually_verified'] == 0].index  # 噪声数据索引

# print(f"干净数据数量: {len(clean_idx)}")
# print(f"噪声数据数量: {len(noisy_idx)}")

# # 3. 提取噪声样本ID (从文件名提取数字部分)
# noisy_ids = [int(fname.split('.')[0]) for fname in train_csv.loc[noisy_idx, 'fname']]
# print(f'\nnoisy ids:{noisy_ids}')

# clean_ids = [int(fname.split('.')[0]) for fname in train_csv.loc[clean_idx, 'fname']]

###############


# %%


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
print(f"标签映射示例: {test_label_to_int}")
print(f"前5个文件标签: {list(test_file_to_label.items())[:5]}")

test_file_to_int

# %%


# %%
# 测试读取特征目录的文件

import os
import utils
# 设置特征目录
feature_dir = params_paths['test_feature_extracted']

# 获取前5个特征文件
data_files = [f for f in os.listdir(feature_dir) if f.endswith('_mel.data')][:20]

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

# %%
# Mel 频谱图的生成
import os
import numpy as np
from tqdm import tqdm
from feat_ext import load_audio_file, modify_file_variable_length, get_mel_spectrogram
import utils

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
        print("所有特征文件已存在，无需处理")
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

# %%
# 如果需要训练所有数据，需要将train_data设置为'all'

from data import get_label_files
from sklearn.model_selection import train_test_split
# 初始化数据生成器
feature_dir = params_paths['train_feature_extracted']

if params_ctrl.get('train_data') == 'all':
    file_list = [f for f in os.listdir(feature_dir) if f.endswith('_mel.data')]
if params_ctrl.get('train_data') == 'clean':
    file_list = [filelist_audio_train[i].replace('.wav', suffix_in + '.data') for i in clean_set]

print(f"文件列表:{file_list},\n 特征文件数量: {len(file_list)}\n")

###


# 初始化数据生成器
# get label for every file *from the .data saved in disk*, in float
labels_audio_train = get_label_files(filelist=file_list,
                                     dire=params_paths.get('train_feature_extracted'),
                                     suffix_in=suffix_in,
                                     suffix_out=suffix_out
                                     )

# sanity check
print('Number of clips considered as train set: {0}'.format(len(file_list)))
print('Number of labels loaded for train set: {0}'.format(len(labels_audio_train)))

# split the val set randomly (but stratified) within the train set
train_files, val_files = train_test_split(file_list,
                                       test_size=params_learn.get('val_split'),
                                       stratify=labels_audio_train,
                                       random_state=42
                                       )

###
# 准备数据 load data
# import numpy as np
from data import DataGeneratorPatch
from torch.utils.data import Dataset, DataLoader
import torch
import sklearn

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



train_data_gen = DataGeneratorPatch(
    feature_dir=feature_dir,
    file_list=train_files,
    params_learn=params_learn,
    params_extract=params_extract,
    suffix_in='_mel',
    suffix_out='_label'
)
val_data_gen = DataGeneratorPatch(
    feature_dir=feature_dir,
    file_list=val_files,
    params_learn=params_learn,
    params_extract=params_extract,
    suffix_in='_mel',
    suffix_out='_label',
    scaler=train_data_gen.scaler
)

# 创建PyTorch兼容的数据集
train_dataset = TorchDataWrapper(train_data_gen)
val_dataset = TorchDataWrapper(val_data_gen)

# 创建PyTorch DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=None,  # 因为原作者已处理批次
    shuffle=True,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=None,  # 因为原作者已处理批次
    shuffle=False,
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

for batch_idx, (features, labels) in enumerate(train_loader):
    print(f"\nPyTorch DataLoader 第一个batch:")
    print(f"特征张量形状: {features.shape}")  # [batch, 1, time, freq]
    print(f"标签张量形状: {labels.shape}")    # [batch]
    for i in range(features.shape[0]):
        print(f"\n样本{i} 标签: {labels[i].item()}")
        print(f"样本{i} 特征片段:\n{features[i, 0, :5, :5]}")
    break


# %%
from baseline_cnn import BaselineCNN
from torch import nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("BaselineCNN imported successfully!")
model = BaselineCNN(
    n_mels=params_extract['n_mels'],
    patch_len=params_extract['patch_len'],
    n_classes=params_learn['n_classes']
).to(device)

# 添加模型权重加载功能
pretrained_path = params_paths['pretrained']  # 替换为你的预训练模型路径
if os.path.exists(pretrained_path):
    model.load_state_dict(torch.load(pretrained_path))
    print('成功加载预训练权重')
else:
    print('未找到预训练模型，将从零开始训练')


print('loading optimizer and loss function...')
# 定义优化器和损失函数
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
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
    print(f'从检查点恢复训练，epoch={start_epoch}, 最佳准确率={best_acc:.4f}')


# 创建TensorBoard writer
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 训练循环
def train(model, loader, val_loader, optimizer, criterion, epochs=160):
    model.train()
    best_acc = 0.0

    # 添加检查点目录
    os.makedirs('checkpoints', exist_ok=True)

    # 添加epoch进度条
    epoch_pbar = tqdm(range(epochs), desc='Training', unit='epoch')

    for epoch in epoch_pbar:
        try:
            model.train()
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

            ###
            model.eval()
            val_loss = 0
            val_correct = 0

            # 添加batch进度条
            val_batch_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}', leave=False)

            with torch.no_grad():
                print('val_loader: eval model loss')
                for features, labels in val_batch_pbar:
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels.squeeze())
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels.squeeze()).sum().item()

                    val_batch_pbar.set_postfix(loss=loss.item())

            # 计算验证集指标
            val_accuracy = val_correct / len(val_loader.dataset)
            avg_val_loss = val_loss / len(val_loader)

            ###

            # 记录到TensorBoard
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Accuracy/train', accuracy, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch) 
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)

             # 记录到TensorBoard
            # writer.add_scalar('Loss/train', avg_loss, epoch)
            # writer.add_scalar('Accuracy/train', accuracy, epoch)

            # 更新学习率
            scheduler.step(val_accuracy)

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
train(model, train_loader, val_loader, optimizer, criterion)


assert 1 == 2 
# %%
# evaluate the model 用测试集测试训练好的模型
from data import PatchGeneratorPerFile
from baseline_cnn import BaselineCNN
from scipy.stats.mstats import gmean
model = BaselineCNN(
    n_mels=params_extract['n_mels'],
    patch_len=params_extract['patch_len'],
    n_classes=params_learn['n_classes']
)

# model.load_state_dict(torch.load('test-overfit1.pth', map_location=torch.device('cpu')))
# 修改模型加载方式，添加map_location参数
model.load_state_dict(torch.load('/Users/nadinebrewington/Desktop/wav2vec/icassp19/wholebatch418.pth', map_location=torch.device('cpu')))

test_files_list = [f for f in os.listdir(params_paths.get('test_feature_extracted')) if f.endswith(suffix_in + '.data')]
test_files_list

test_preds = np.empty((len(test_files_list), params_learn.get('n_classes')))



test_gen_patch = DataGeneratorPatch(
    feature_dir=params_paths.get('test_feature_extracted'),
    file_list=test_files_list,
    params_extract=params_extract,
    params_learn=params_learn,

    suffix_in='_mel',
    floatx=np.float32,
    scaler=train_data_gen.scaler
    )
test_dataset = TorchDataWrapper(test_gen_patch)

eval_loader = DataLoader(
    test_dataset,
    batch_size=None,  # 因为原作者已处理批次
    shuffle=False,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)


model.eval()

# test_preds = np.empty((len(test_files_list), params_learn['n_classes']))
###
model.eval()
eval_loss = 0
eval_correct = 0

# 添加batch进度条
eval_batch_pbar = tqdm(eval_loader, desc=f'Eval the model', leave=False)

with torch.no_grad():
    print('val_loader: eval model loss')
    for features, labels in eval_batch_pbar:
        features = features.to(device)
        labels = labels.to(device)
        outputs = model(features)
        # loss = criterion(outputs, labels.squeeze())
        # eval_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        eval_correct += (predicted == labels.squeeze()).sum().item()

        eval_batch_pbar.set_postfix()

# 计算验证集指标
eval_accuracy = eval_correct / len(eval_loader.dataset)
avg_eval_loss = eval_loss / len(eval_loader)

print(f'Validation Accuracy: {eval_accuracy:.4f}') 

# 获取最终预测标签
# pred_labels = np.argmax(test_preds, axis=1)



# 打印结果时显示真实标签
# print("\n测试结果样例 (文件名, 预测标签, 真实标签):")
# for i in range(min(20, len(test_files_list))):
#     true_label = int(test_labels[i][0])  # 获取真实标签的整数值
#     if hasattr(train_data_gen, 'int_to_label'):
#         true_label_name = train_data_gen.int_to_label[true_label]  # 转换为标签名
#     else:
#         true_label_name = str(true_label)
    
#     print(f"文件: {test_files_list[i]}, 预测标签: {pred_labels[i]}, 真实标签: {true_label_name}")

# 打印部分结果
# print("\n测试结果样例:")
# for i in range(min(20, len(test_files_list))):
#     print(f"文件: {test_files_list[i]}, 预测标签: {pred_labels[i]}")




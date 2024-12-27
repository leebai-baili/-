import os
import yaml
import numpy as np
import torch
import cv2
from mydata import MyDataset, get_valid_transforms
from seq2seq import Generator
from utils import torch_PSNR, torch_SSIM  # 导入 PSNR 和 SSIM 函数

# 加载配置文件
with open('/home/gem/GuanH/Program/seq2seq/config_examples/my_data.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 初始化模型
net = Generator(config, device= 'cuda:6')
net.cuda()
net.load_state_dict(torch.load('/home/gem/GuanH/Program/ckpt1/ckpt_best.pth'))

# 数据路径
data_path = r'/home/gem/GuanH/Program/Data/dault_image_npy'
data_list = os.listdir(data_path)[:2000]
valid_data = MyDataset(data_path, data_list, get_valid_transforms())

# 创建结果保存路径
result_dir = 'result_images/'
os.makedirs(result_dir, exist_ok=True)

# 验证阶段
psnr_scores = []
ssim_scores = []

# 遍历验证集
for idx in range(len(valid_data)):
    # 加载数据
    data = valid_data[idx]
    source_seq = data[0].unsqueeze(0).cuda()  # 输入序列
    target_seq = data[1].unsqueeze(0).cuda()  # 目标序列

    # 生成预测
    c_s = 64
    target_code = torch.ones((1, c_s)).cuda()
    output_target = net(source_seq, target_code, n_outseq=target_seq.shape[1])

    # 转换维度以计算指标
    output_target = output_target.squeeze().permute(0, 3, 2, 1)  # [T, H, W, C] -> [T, C, H, W]
    target_seq = target_seq.squeeze().permute(0, 3, 2, 1)

    # 计算 PSNR 和 SSIM
    psnr = torch_PSNR(target_seq, output_target, data_range=1.0)
    ssim = torch_SSIM(target_seq, output_target, data_range=1.0)

    psnr_scores.append(psnr.item())
    ssim_scores.append(ssim.item())

    # 保存预测图像
    for i in range(output_target.shape[0]):
        predict = output_target[i].detach().cpu().numpy() * 255.0  # 转为像素值
        predict = predict.astype(np.uint8)
        cv2.imwrite(os.path.join(result_dir, f'pred_{idx}_{i}.jpg'), predict)

    # 保存目标图像
    for i in range(target_seq.shape[0]):
        target = target_seq[i].detach().cpu().numpy() * 255.0  # 转为像素值
        target = target.astype(np.uint8)
        cv2.imwrite(os.path.join(result_dir, f'target_{idx}_{i}.jpg'), target)

# 输出平均 PSNR 和 SSIM
average_psnr = np.mean(psnr_scores)
average_ssim = np.mean(ssim_scores)

print(f"Average PSNR: {average_psnr:.4f}")
print(f"Average SSIM: {average_ssim:.4f}")

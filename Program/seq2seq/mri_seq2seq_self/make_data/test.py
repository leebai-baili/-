import torch
print(torch.cuda.is_available())  # 是否支持 CUDA
print(torch.version.cuda)         # 使用的 CUDA 版本
print(torch.cuda.get_device_name(6))  # 查看 GPU 名称

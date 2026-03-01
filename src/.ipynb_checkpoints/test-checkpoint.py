import torch
import torch.nn as nn

# 模拟你的窗口/步长配置
win_size = (12, 12)
stride = (12, 12)
x = torch.randn(2, 3, 24, 24).cuda()  # 你的输入形状

# 旧版本bug验证
unfold = nn.Unfold(kernel_size=win_size, stride=stride).cuda()
x_unfold = unfold(x)
x_blocks = x_unfold.transpose(1,2).reshape(2, 4, 3, 12, 12).permute(0,2,3,4,1)

# 验证元素总数是否正确（2*3*24*24 = 3456；2*3*12*12*4=3456）
print(f"输入元素总数: {x.numel()}")
print(f"blocks元素总数: {x_blocks.numel()}")
# 若总数不等 → 触发形状计算bug；若总数相等但元素值错 → 触发非连续/维度错位bug

# 验证元素值一致性（取第一个窗口块）
x_first_win = x[:, :, 0:12, 0:12]  # 预期的第一个窗口
x_blocks_first = x_blocks[:, :, :, :, 0]  # Unfold后的第一个窗口
print(f"元素值是否一致: {torch.allclose(x_first_win, x_blocks_first, atol=1e-5)}")
# 旧版本会返回False，新版本返回True
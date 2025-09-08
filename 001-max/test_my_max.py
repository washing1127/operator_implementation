import torch
import numpy as np
from my_max import my_max

# 测试一维张量
def test_1d_tensor():
    print('测试一维张量')
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 测试整个张量的最大值
    result = my_max(a, dim=None)
    print(f'整个张量的最大值: {result}')
    assert result == 5.0
    
    # 测试指定维度的最大值
    result = my_max(a, dim=0)
    print(f'dim=0的最大值: {result}')
    assert result == 5.0
    
    # 测试keepdim=True
    result = my_max(a, dim=0, keepdim=True)
    print(f'dim=0, keepdim=True的最大值: {result}')
    assert result.shape == (1,)
    assert result.item() == 5.0

# 测试二维张量
def test_2d_tensor():
    print('\n测试二维张量')
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f'原始张量:\n{a}')
    
    # 测试整个张量的最大值
    result = my_max(a, dim=None)
    print(f'整个张量的最大值: {result}')
    assert result == 4.0
    
    # 测试dim=0的最大值
    result = my_max(a, dim=0)
    print(f'dim=0的最大值: {result}')
    assert torch.all(result == torch.tensor([3.0, 4.0]))
    
    # 测试dim=1的最大值
    result = my_max(a, dim=1)
    print(f'dim=1的最大值: {result}')
    assert torch.all(result == torch.tensor([2.0, 4.0]))
    
    # 测试keepdim=True
    result = my_max(a, dim=1, keepdim=True)
    print(f'dim=1, keepdim=True的最大值: {result}')
    assert result.shape == (2, 1)
    assert torch.all(result == torch.tensor([[2.0], [4.0]]))

# 测试三维张量
def test_3d_tensor():
    print('\n测试三维张量')
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    print(f'原始张量形状: {a.shape}')
    
    # 测试整个张量的最大值
    result = my_max(a, dim=None)
    print(f'整个张量的最大值: {result}')
    assert result == 8.0
    
    # 测试dim=0的最大值
    result = my_max(a, dim=0)
    print(f'dim=0的最大值形状: {result.shape}')
    assert result.shape == (2, 2)
    
    # 测试dim=1的最大值
    result = my_max(a, dim=1)
    print(f'dim=1的最大值形状: {result.shape}')
    assert result.shape == (2, 2)
    
    # 测试dim=2的最大值
    result = my_max(a, dim=2)
    print(f'dim=2的最大值形状: {result.shape}')
    assert result.shape == (2, 2)
    
    # 测试负数维度
    result = my_max(a, dim=-1)
    print(f'dim=-1的最大值形状: {result.shape}')
    assert result.shape == (2, 2)
    assert torch.all(result == my_max(a, dim=2))

# 运行测试
if __name__ == '__main__':
    test_1d_tensor()
    test_2d_tensor()
    test_3d_tensor()
    print('\n所有测试通过！')


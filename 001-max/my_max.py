from typing import Optional
import torch

def my_max(input: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False):
    # 如果dim为None，则计算整个张量的最大值
    if dim is None:
        # 将张量展平为一维
        flattened = input.flatten()
        max_val = flattened[0]
        for i in range(1, flattened.size(0)):
            if flattened[i] > max_val:
                max_val = flattened[i]
        return max_val
    
    # 获取输入张量的形状
    input_shape = input.shape
    
    # 计算指定维度的最大值
    if dim < 0:
        dim = len(input_shape) + dim
    
    # 创建输出张量的形状
    if keepdim:
        output_shape = list(input_shape)
        output_shape[dim] = 1
    else:
        output_shape = list(input_shape)
        output_shape.pop(dim)
    
    # 重新排列维度，将目标维度放到最后
    perm = list(range(len(input_shape)))
    perm.pop(dim)
    perm.append(dim)
    permuted = input.permute(perm)
    
    # 重塑张量，将目标维度展平
    reshaped = permuted.reshape(-1, input_shape[dim])
    
    # 计算每行的最大值
    result = torch.empty(reshaped.shape[0], dtype=input.dtype, device=input.device)
    for i in range(reshaped.shape[0]):
        max_val = reshaped[i][0]
        for j in range(1, reshaped.shape[1]):
            if reshaped[i][j] > max_val:
                max_val = reshaped[i][j]
        result[i] = max_val
    
    return result.reshape(output_shape)


if __name__ == '__main__':
    a = torch.tensor([
        [
            [1, 3, 7, 4],
            [2, 8, 6, 9],
            [4, 2, 7, 7]
        ],
        [
            [9, 6, 5, 2],
            [3, 4, 7, 6],
            [1, 2, 3, 4]
        ]
                      ])
    print(my_max(a, dim=1))
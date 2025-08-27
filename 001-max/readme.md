功能参考 `torch.max`，设置 `input`，`dim` 和 `keepdim` 三个参数。

- `input: torch.Tensor` 表示输入的张量；
- `dim: Optional[int]` 表示要消减的维度，默认为 `None` 表示消减所有维度；
- `keepdim: Optional[bool]` 若为 `True`，则输出维度保持和输入维度一致，`dim` 维为 `1`，默认为 `False`；

返回值为对应维度的最大值。

示例：
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
        [ 1.1949, -1.1127, -2.2379, -0.6702],
        [ 1.5717, -0.9207,  0.1297, -1.8768],
        [-0.6172,  1.0036, -0.6060, -0.2432]])
>>> my_max(a, 1)
tensor([0.8475, 1.1949, 1.5717, 1.0036])
>>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
>>> my_max(a, dim=1, keepdim=True)
tensor([[2.], [4.]])
>>> my_max(a, dim=1, keepdim=False)
values=tensor([2., 4.])
```
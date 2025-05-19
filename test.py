import torch

# 產生介於 -2 到 2 的 3x3 tensor
tensor = (torch.rand(3, 3) * 4) - 2  # rand() 產生 [0,1)，*4 得到 [0,4)，-2 得到 [-2,2)
print("Tensor:")
print(tensor)

# 判斷
result = tensor > 1
print("Tensor > 1:")
print(result)

import torch

tensor0d = torch.tensor(1)
tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype)

tensor2d = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
    ]
)
tensor3d = torch.tensor(
    [
        [
            [1, 2],
            [3, 4],
        ],
        [
            [5, 6],
            [7, 8],
        ],
    ]
)

floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)

floatvec = tensor1d.to(torch.float32)
print(floatvec.dtype)

print(tensor2d)
print(tensor2d.shape)

print(tensor2d.reshape(3, 2))
print(tensor2d.view(3, 2))

print(tensor2d.T)

print(tensor2d.matmul(tensor2d.T))
print(tensor2d @ tensor2d.T)

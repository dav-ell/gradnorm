# gradnorm

There are several other PyTorch implementations of GradNorm:
 1. https://github.com/lucidrains/gradnorm-pytorch
 2. https://github.com/LucasBoTang/GradNorm
 3. https://github.com/brianlan/pytorch-grad-norm
 4. https://github.com/hosseinshn/GradNorm

However, each of these has one or more shortcomings that can cause issues when using them in larger PyTorch training frameworks:
 1. The library requires delegating execution of `.backward()` to it, preventing frameworks like PyTorch Lightning or Keras from being able to call the `.backward()`, as in [1](https://github.com/lucidrains/gradnorm-pytorch) and [2](https://github.com/LucasBoTang/GradNorm).
 2. The library is not standalone, as in [3](https://github.com/brianlan/pytorch-grad-norm).
 3. 



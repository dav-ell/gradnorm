# gradnorm

Simple, easy-to-integrate PyTorch implementation of GradNorm: https://arxiv.org/abs/1711.02257.

---

Despite its popularity, GradNorm has rather few mature implementations in open source. There are several other PyTorch implementations of GradNorm:
 1. https://github.com/lucidrains/gradnorm-pytorch
 2. https://github.com/LucasBoTang/GradNorm
 3. https://github.com/brianlan/pytorch-grad-norm
 4. https://github.com/hosseinshn/GradNorm

However, each of these has one or more shortcomings that can cause issues when using them in larger PyTorch training frameworks:
 1. The library requires delegating execution of `.backward()` to it, preventing frameworks like PyTorch Lightning or Keras from being able to call the `.backward()`, as in [1](https://github.com/lucidrains/gradnorm-pytorch) and [2](https://github.com/LucasBoTang/GradNorm).
 2. The library is not standalone, as in [3](https://github.com/brianlan/pytorch-grad-norm).
 3. The library lacks verifiably correct behavior compared to the original paper, as in [1], [3]-[4].

This repo aims to address this, by being completely separate from the model's training loop and having verified performance.

# Usage

Usage is easy, just 3 lines:
```python
gradnorm = GradNorm(model.last_shared_layer, alpha=0.12, number_of_tasks=10,
                    lr=1e-3, lr_warmup=1000, device='cuda:0')

# in training loop
L_grad = gradnorm.gradnorm(task_losses)
gradnorm.apply_grads(L_grad)
```

Insert it into your training loop like so:
```python
for epoch in range(epochs):
    for x, y_true in dataloader:
        x, y_true = x.to(device), y_true.to(device)

        y_pred = model(x)

        # Your training loop's loss calculation for each task
        task_losses = F.mse_loss(y_pred, y_true, reduction='none').mean(dim=-1)
        L_i = task_losses.mean(dim=0)

        L = torch.sum(gradnorm.w_i * L_i)  # <--- Use gradnorm weights w_i here

        # Your training loop's backward call
        optimizer.zero_grad()
        L.backward(retain_graph=True)  # retain_graph=True is currently required

        # Compute the GradNorm loss
        L_grad = gradnorm.gradnorm(L_i)  # <--- Line 1
        gradnorm.apply_grads(L_grad)  # <--- Line 2

        optimizer.step()
```

# Performance

We re-create the toy model experiment from the paper with the below results.

<img height="400" alt="image" src="https://github.com/dav-ell/gradnorm/assets/17986781/44d25e11-84a5-423d-8e49-946988e0fd36">

Where the paper results are:

<img height="300" alt="image" src="https://github.com/dav-ell/gradnorm/assets/17986781/a84225f4-3d9e-4fb4-a6b9-cb639fcb8383">

Experimenting with different learning rates yields slightly different curves. Adding a learning rate warmup to the toy model training yields even smoother weight adjustments.

<img height="400" alt="image" src="https://github.com/dav-ell/gradnorm/assets/17986781/f0be12e4-0f44-4f34-b531-85ec48380236">


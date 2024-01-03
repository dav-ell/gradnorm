import torch
from torch import nn
import torch.nn.functional as F

class GradNorm:
    """
    GradNorm implementation designed for maximal compatibility with PyTorch training frameworks.

    API for this GradNorm implementation:
        1. Initialize the GradNorm class with the model, alpha, and number of tasks
        2. Compute your task losses, as you would normally, store in a tensor of shape [T]
        3. Apply gradnorm, passing losses as input; w_i updated automatically
        4. Perform backpropagation to your model as usual
    """
    def __init__(self, layer: nn.Module, alpha: float, number_of_tasks: int, lr: float = None, lr_warmup: float = None, device: str = "cpu"):
        """
        Initialize the GradNorm class.
        
        :param layer: The multitask learning layer shared by all tasks.
        :param alpha: The GradNorm alpha parameter, higher if tasks are more different.
        :param number_of_tasks: Number of tasks in the multitask learning model.
        """
        self.layer = layer
        self.alpha = alpha
        self.T = number_of_tasks
        self.device = torch.device(device)
        self.w_i = torch.nn.Parameter(torch.ones(self.T, device=self.device), requires_grad=True) # Step 1: Initialize task weights
        self.L_i_0 = None  # Placeholder for the initial losses
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.warmup_step = 1

    def gradnorm(self, L_i: torch.Tensor, layer: nn.Module = None) -> torch.Tensor:
        """
        Compute the GradNorm loss.
        
        :param task_losses: A tensor of losses, one for each task.
        :return: The GradNorm loss.
        """

        if layer is None:
            layer = self.layer
        
        assert layer is not None and isinstance(layer, nn.Module), "Must provide a layer to compute the GradNorm loss."
        
        # Step 2: Save the initial losses for each task if not already saved
        if self.L_i_0 is None:
            self.L_i_0 = L_i.detach()

        # Step 3: Compute gradient norms for each task and the average gradient norm
        G_W_i = torch.stack([
            torch.autograd.grad(L_i[i] * self.w_i[i], layer.parameters(), retain_graph=True,
                                create_graph=True)[0].norm()
            for i in range(self.T)])
        G_W_bar = torch.mean(G_W_i)

        # Step 4: Compute relative inverse training rates r_i(t)
        tilde_L_i = L_i / self.L_i_0
        r_i = tilde_L_i / torch.mean(tilde_L_i)

        # Step 5: Calculate the GradNorm loss L_grad
        target_G_W_i = (G_W_bar * torch.pow(r_i, self.alpha)).detach()
        L_grad = F.l1_loss(G_W_i, target_G_W_i)

        return L_grad

    def apply_grads(self, L_grad: torch.Tensor, lr: float = None) -> torch.Tensor:
        """
        Apply the gradients from the GradNorm loss and the total loss.
        
        :param optimizer: The optimizer for the model parameters.
        :param lr: Optional learning rate for updating task weights.
        :return: The updated task weights.
        """

        if lr is None:
            lr = self.lr

            if self.lr_warmup is not None:
                lr = lr * min(1., float(self.warmup_step) / self.lr_warmup)
                self.warmup_step += 1

        assert lr is not None, "Must provide a learning rate to apply_grads."

        # Step 6: Differentiate L_grad with respect to task weights w_i and update
        self.w_i.grad = torch.autograd.grad(L_grad, self.w_i)[0]
        self.w_i.data -= lr * self.w_i.grad

        # # Step 7: Renormalize task weights w_i
        self.w_i.data = self.w_i / torch.sum(self.w_i) * self.T

        if torch.any(self.w_i < 0):
            print("Negative w_i values detected. Consider reducing the gradnorm learning rate.")
            self.w_i.data = torch.clamp(self.w_i.data, min=1e-8)

        return self.w_i

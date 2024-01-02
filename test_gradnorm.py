import unittest
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl

pl.seed_everything(42)

class ToyModel(nn.Module):
    def __init__(self, T, input_dim=250, hidden_dim=100, output_dim=100):
        super().__init__()
        # Common trunk: 4-layer fully-connected ReLU-activated network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        # Final affine transformation layer
        self.final = nn.Linear(hidden_dim, T * output_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        return self.final(x).view(x.size(0), -1, 100)  # Reshape output for T tasks

    @property
    def last_shared_layer(self):
        return self.fc4

class GradNorm:
    def __init__(self, layer, alpha, number_of_tasks):
        """
        Initialize the GradNorm class.
        
        :param layer: The multitask learning layer.
        :param alpha: The hyperparameter from the GradNorm algorithm.
        :param number_of_tasks: Number of tasks in the multitask learning model.
        """
        self.layer = layer
        self.alpha = alpha
        self.T = number_of_tasks
        self.w_i = torch.nn.Parameter(torch.ones(self.T), requires_grad=True)  # Step 1: Initialize task weights
        self.L_i_0 = None  # Placeholder for the initial losses

    def compute_grad_norm_loss(self, L_i):
        """
        Compute the GradNorm loss.
        
        :param L_i: A tensor of losses, one for each task.
        :return: The GradNorm loss.
        """
        # Step 2: Save the initial losses for each task if not already saved
        if self.L_i_0 is None:
            self.L_i_0 = L_i.detach()

        # Step 3: Compute gradient norms for each task and the average gradient norm
        G_W_i = torch.stack([
            torch.autograd.grad(L_i[i] * self.w_i[i], self.layer.parameters(), retain_graph=True,
                                create_graph=True, only_inputs=True)[0].norm()
            for i in range(self.T)])
        G_W_bar = torch.mean(G_W_i)

        # Step 4: Compute relative inverse training rates r_i(t)
        tilde_L_i = L_i / self.L_i_0
        r_i = tilde_L_i / torch.mean(tilde_L_i)

        # Step 5: Calculate the GradNorm loss L_grad
        target_G_W_i = (G_W_bar * torch.pow(r_i, self.alpha)).detach()
        L_grad = F.l1_loss(G_W_i, target_G_W_i)

        return L_grad

    def apply_grads(self, optimizer, L_grad):
        """
        Apply the gradients from the GradNorm loss and the total loss.
        
        :param optimizer: The optimizer for the model parameters.
        :param L_grad: The GradNorm loss.
        """

        # Step 6: Differentiate L_grad with respect to task weights w_i and update
        self.w_i.grad = torch.autograd.grad(L_grad, self.w_i)[0]
        self.w_i.data -= optimizer.param_groups[0]['lr'] * self.w_i.grad

        # Step 7: Renormalize task weights w_i
        self.w_i.data *= self.T / torch.sum(self.w_i)

        if torch.any(self.w_i < 0):
            import code; code.interact(local=dict(globals(), **locals()))

        return self.w_i


class TestGradNorm(unittest.TestCase):

    def setUp(self):
        # Assuming T=2 for simplicity
        self.T = 10
        self.model = ToyModel(T=self.T)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.alpha = 0.12
        self.grad_norm = GradNorm(self.model.last_shared_layer, alpha=self.alpha, number_of_tasks=self.T)
        self.w_i_history = []  # Store w_i values at each step

    def test_gradnorm(self):
        # Define the scaling factors for each task
        sigma = torch.rand(self.T) * 1000.0

        # Dummy input and output (adjust as needed)
        x = torch.randn(32, 250)  # Batch size 32, input dimension 250
        y_true = torch.randn(32, self.T, 100)  # Corresponding outputs for T=2 tasks

        # Train the model using the GradNorm class
        for step in range(1000):  # Number of training steps
            self.optimizer.zero_grad()

            y_pred = self.model(x)

            # Apply scaling factors to the outputs for each task
            scaled_y_pred = torch.stack([sigma[i] * y_pred[:, i] for i in range(self.T)], dim=1)
            # Calculate losses for each task
            task_losses = F.mse_loss(scaled_y_pred, y_true, reduction='none').mean(dim=-1)
            L_i = task_losses.mean(dim=0)

            # Compute the GradNorm loss
            L_grad = self.grad_norm.compute_grad_norm_loss(L_i)

            # Apply gradients from the GradNorm loss and the total loss
            L = torch.sum(self.grad_norm.w_i * L_i)                
            self.grad_norm.apply_grads(self.optimizer, L_grad)

            # Step 8: Compute standard gradients for network weights and update
            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()

            # Store w_i at each step
            self.w_i_history.append(self.grad_norm.w_i.clone().detach())

            # Log progress (optional, for monitoring during training)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {L.item()}, L_grad: {L_grad.item()}, w_i: {self.grad_norm.w_i.data}")

        # Plot w_i values
        self.plot_w_i_history()

    def plot_w_i_history(self):
        w_i_history = torch.stack(self.w_i_history)
        plt.figure()
        for i in range(self.T):
            plt.plot(w_i_history[:, i].numpy(), label=f'w_{i+1}')
        plt.xlabel('Step')
        plt.ylabel('w_i')
        plt.legend()
        plt.savefig('w_i_history.png')


if __name__ == '__main__':
    unittest.main()
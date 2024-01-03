import unittest
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from gradnorm import GradNorm


pl.seed_everything(135)


class ToyModel(nn.Module):
    def __init__(self, T, input_dim=250, hidden_dim=100, output_dim=100):
        super().__init__()
        # Common trunk: 4-layer fully-connected ReLU-activated network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        # Final affine transformation layer
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(T)])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return torch.stack([head(x) for head in self.heads], dim=1)

    @property
    def last_shared_layer(self):
        return self.fc4


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, num_tasks=10, sigmas=[48, 3, 54, 16, 9, 30, 52, 26, 47, 81], num_samples=10000, input_dim=250, hidden_dim=100):
        super().__init__()
        
        # Dataset
        self.sigmas = torch.tensor(sigmas).float()  # Scaling factors for each task
        B = torch.normal(mean=0., std=10., size=(input_dim, hidden_dim))
        epsilons = torch.normal(mean=0., std=3.5, size=(len(self.sigmas), input_dim, hidden_dim))
        self.x = torch.rand(num_samples, input_dim)
        self.ys = torch.stack([self.sigmas[i] * torch.tanh(self.x @ (B + epsilons[i])) for i in range(len(self.sigmas))], axis=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.ys[idx]


class TestGradNorm(unittest.TestCase):

    def setUp(self):
        self.device = 'mps'

        self.T = 2
        self.model = ToyModel(T=self.T).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.gradnorm = GradNorm(self.model.last_shared_layer, alpha=0.12, number_of_tasks=self.T, lr=1e-3, device=self.device)
        self.w_i_history = []  # Store w_i values at each step

        # Toy data
        batch_size = 100
        self.dataset = ToyDataset(num_tasks=self.T, sigmas=[1, 100])
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def test_gradnorm(self):        

        for epoch in range(250):
            for step, (x, y_true) in enumerate(self.dataloader):
                x, y_true = x.to(self.device), y_true.to(self.device)

                y_pred = self.model(x)

                # Calculate losses for each task
                task_losses = F.mse_loss(y_pred, y_true, reduction='none').mean(dim=-1)
                L_i = task_losses.mean(dim=0)

                L = torch.sum(self.gradnorm.w_i * L_i)  
                self.optimizer.zero_grad()
                L.backward(retain_graph=True)

                # Compute the GradNorm loss
                L_grad = self.gradnorm.gradnorm(L_i)

                # Apply gradients from the GradNorm loss and the total loss
                self.gradnorm.apply_grads(L_grad)

                self.optimizer.step()

                # Store w_i at each step
                self.w_i_history.append(self.gradnorm.w_i.clone().detach())

            # Log progress (optional, for monitoring during training)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {L.item()}, L_grad: {L_grad.item()}, w_i: {self.gradnorm.w_i.data}")

        # Plot w_i values
        self.plot_w_i_history()

    def plot_w_i_history(self):
        w_i_history = torch.stack(self.w_i_history)
        
        plt.figure(figsize=(16, 12))
        for i in range(self.T):
            plt.plot(w_i_history[:, i].numpy(), lw=3, label=f'σ = {self.dataset.sigmas[i]}')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(0, 28000)
        plt.xlabel("Iters", fontsize=18)
        plt.ylabel("w_i", fontsize=18)
        plt.title("Adaptive Weights During Training for Each Task", fontsize=18)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('w_i_history.png')


if __name__ == '__main__':
    unittest.main()

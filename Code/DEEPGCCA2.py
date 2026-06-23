"""
DEEPGCCA2.py
Corrected DeepGCCA with independent per-view projection matrices and
pure cross-view covariance loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CorrectedDeepGCCA(nn.Module):
    def __init__(self, view_dims, hidden_dim=64, common_dim=12):
        super().__init__()
        self.num_views = len(view_dims)
        self.hidden_dim = hidden_dim
        self.common_dim = common_dim

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for dim in view_dims
        ])

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, dim),
                nn.Sigmoid()
            ) for dim in view_dims
        ])

        self.weight_params = nn.Parameter(torch.ones(self.num_views) / self.num_views)
        self.temp = nn.Parameter(torch.tensor(1.0))

        self.U_list = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, common_dim)) for _ in range(self.num_views)
        ])
        for U in self.U_list:
            with torch.no_grad():
                Q, R = torch.qr(U)
                d = torch.diag(R)
                Q = Q * d.sign().unsqueeze(0)
                U.copy_(Q)

    def _orthogonalize(self, U):
        with torch.no_grad():
            Q, R = torch.qr(U)
            d = torch.diag(R)
            Q = Q * d.sign().unsqueeze(0)
            U.copy_(Q)

    def forward(self, views):
        batch_size = views[0].shape[0]

        z_list = [encoder(view) for encoder, view in zip(self.encoders, views)]
        recon_list = [decoder(z) for decoder, z in zip(self.decoders, z_list)]

        z_centered = [z - z.mean(dim=0, keepdim=True) for z in z_list]
        z_normalized = [z_c / (torch.norm(z_c, dim=1, keepdim=True) + 1e-8) for z_c in z_centered]

        weights = F.softmax(self.weight_params / self.temp, dim=0)

        for U in self.U_list:
            self._orthogonalize(U)

        h_list = [z_n @ U for z_n, U in zip(z_normalized, self.U_list)]

        weighted_h = [weights[i] * h_list[i] for i in range(self.num_views)]
        fused = torch.stack(weighted_h).sum(dim=0)

        recon_loss = sum(F.mse_loss(recon, view) for recon, view in zip(recon_list, views)) / self.num_views

        weighted_corr = torch.zeros(self.hidden_dim, self.hidden_dim, device=device)
        for i in range(self.num_views):
            for j in range(self.num_views):
                if i == j:
                    continue
                cov_ij = z_normalized[i].T @ z_normalized[j] / batch_size
                weighted_corr += weights[i] * weights[j] * cov_ij
        gcca_loss = (self.hidden_dim - torch.trace(weighted_corr)) / self.hidden_dim

        ortho_loss = 0.0
        for U in self.U_list:
            ortho_loss += torch.norm(U.T @ U - torch.eye(self.common_dim, device=device), p='fro') ** 2
        ortho_loss = ortho_loss / self.num_views

        entropy_loss = -torch.sum(weights * torch.log(weights + 1e-8))

        lambda1 = 1.0    # recon
        lambda2 = 0.1    # gcca
        lambda3 = 0.0    # ortho (reserved)
        lambda4 = 0.1    # entropy
        lambda_sum = lambda1 + lambda2 + lambda3 + lambda4

        total_loss = (lambda1 / lambda_sum) * recon_loss + \
                     (lambda2 / lambda_sum) * gcca_loss + \
                     (lambda3 / lambda_sum) * ortho_loss + \
                     (lambda4 / lambda_sum) * entropy_loss

        return fused, total_loss, recon_loss, gcca_loss, weights


def train_corrected_deepgcca(views_tensor, view_dims, hidden_dim=64, common_dim=12,
                              epochs=200, loss_threshold=0.1, lr=1e-3):
    model = CorrectedDeepGCCA(view_dims, hidden_dim=hidden_dim, common_dim=common_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    print(f"Training Corrected DeepGCCA (max {epochs} epochs, threshold {loss_threshold})")

    best_loss = float('inf')
    best_weights = None
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        fused, total_loss, recon_loss, gcca_loss, weights = model(views_tensor)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_weights = weights.detach().cpu().numpy()
            best_model_state = model.state_dict().copy()

        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            weights_str = ', '.join([f'{w:.3f}' for w in weights.detach().cpu().numpy()])
            print(f"  Epoch {epoch:03d}/{epochs}: Total={total_loss.item():.6f}, "
                  f"Recon={recon_loss.item():.6f}, GCCA={gcca_loss.item():.6f}, LR={current_lr:.6f}")
            print(f"  Weights: [{weights_str}]")

        if total_loss.item() < loss_threshold:
            print(f"Loss below threshold {loss_threshold}, early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Training finished. Best loss: {best_loss:.6f}")
    print(f"  Best weights: {best_weights}")

    model.eval()
    with torch.no_grad():
        fused, _, _, _, final_weights = model(views_tensor)
        fused_np = fused.cpu().numpy()
        final_weights = final_weights.cpu().numpy()

    return fused_np, final_weights, model

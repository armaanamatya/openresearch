"""
Model architectures for Adam paper reproduction (arXiv:1412.6980).

Implements:
  - LogisticRegression   — Section 6.1: L2-regularized multi-class logistic regression
  - MLP1000              — Section 6.2: 2 × 1000-ReLU + dropout MLP
  - CIFAR10CNN           — Section 6.3: c64-c64-c128-1000 ConvNet
  - VAE                  — Section 6.4: Variational Autoencoder for bias-correction study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Section 6.1: L2-regularized Logistic Regression ────────────────────────

class LogisticRegression(nn.Module):
    """784-dim input → 10 logits (no hidden layer).

    L2 regularization is handled in the optimizer via weight_decay.
    Input is expected to be a flat vector (B, 784).
    """

    def __init__(self, in_features=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)


# ─── Section 6.1: BoW logistic regression for IMDB ──────────────────────────

class BOWLogisticRegression(nn.Module):
    """Bag-of-words logistic regression for IMDB sparse features.

    Input: (B, n_words) pre-computed BoW vector.
    Dropout is applied to the input during training (50%) per Section 6.1.
    """

    def __init__(self, n_words=10000, num_classes=2, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(n_words, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


# ─── Section 6.2: MLP with 2 × 1000 ReLU hidden layers ─────────────────────

class MLP1000(nn.Module):
    """Two fully-connected hidden layers of 1000 ReLU units each.

    Dropout regularization applied to the input and after each ReLU layer.
    Cross-entropy objective with L2 weight decay (applied at optimizer level).
    Input: flat (B, in_features).
    """

    def __init__(self, in_features=784, hidden_size=1000, num_classes=10,
                 dropout=0.5):
        super().__init__()
        self.dropout_in = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.dropout_in(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


# ─── Section 6.3: CIFAR-10 c64-c64-c128-1000 ConvNet ───────────────────────

class CIFAR10CNN(nn.Module):
    """c64-c64-c128-1000 ConvNet for CIFAR-10 (Section 6.3).

    Three alternating stages of 5×5 convolution + 3×3 max-pooling (stride 2),
    followed by a 1000-unit ReLU fully-connected layer.
    Dropout on the input and on the FC layer.

    For 32×32 input:
      conv1(3→64, 5×5, pad=2) → pool(3×3, s=2) → 64×16×16
      conv2(64→64, 5×5, pad=2) → pool(3×3, s=2) → 64×8×8
      conv3(64→128, 5×5, pad=2) → pool(3×3, s=2) → 128×4×4
      fc(128×4×4=2048 → 1000) → ReLU + dropout
      classifier(1000 → 10)
    """

    def __init__(self, in_channels=3, num_classes=10,
                 dropout_input=0.2, dropout_fc=0.5):
        super().__init__()
        self.dropout_input = nn.Dropout(p=dropout_input)

        # Stage 1: 5×5 conv → 64 filters → max-pool
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2: 5×5 conv → 64 filters → max-pool
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 3: 5×5 conv → 128 filters → max-pool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Fully connected: 128×4×4 → 1000 → 10
        self.fc1 = nn.Linear(128 * 4 * 4, 1000)
        self.dropout_fc = nn.Dropout(p=dropout_fc)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        # Input dropout
        x = self.dropout_input(x)

        # Stage 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Stage 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Stage 3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.classifier(x)


# ─── Section 6.4: Variational Autoencoder ────────────────────────────────────

class VAE(nn.Module):
    """Variational Autoencoder for the bias-correction experiment (Section 6.4).

    Encoder: 784 → 400 → [μ, log σ²] (latent_dim)
    Decoder: latent_dim → 400 → 784 (sigmoid)
    Loss: ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
    """

    def __init__(self, in_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_fc1 = nn.Linear(in_dim, hidden_dim)
        self.enc_mean = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, in_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        return self.enc_mean(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_out(h))

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def elbo_loss(recon_x, x, mu, logvar):
        """ELBO loss: reconstruction (BCE) + KL divergence."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        # Reconstruction: binary cross-entropy
        bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # KL divergence: -0.5 * Σ(1 + logvar - μ² - exp(logvar))
        kld = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
        return (bce + kld) / x.size(0)  # per-sample

# medical_anomaly_detector/vqvae_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv, pinv # For inverse and pseudo-inverse

# --- VQ-VAE Model Components ---
class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True), nn.Conv2d(in_dim, res_h_dim, 3, 1, 1, bias=False),
            nn.ReLU(True), nn.Conv2d(res_h_dim, h_dim, 1, 1, bias=False) )
    def forward(self, x): return x + self.res_block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.stack = nn.ModuleList([ResidualLayer(in_dim,h_dim,res_h_dim) for _ in range(n_res_layers)])
    def forward(self, x):
        for layer in self.stack: x = layer(x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim//2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(h_dim//2, h_dim, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, 3, 1, 1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers) )
    def forward(self, x): return self.conv_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_channels=1):
        super(Decoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, 3, 1, 1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers), nn.ReLU(),
            nn.ConvTranspose2d(h_dim, h_dim//2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, out_channels, 4, 2, 1), nn.Sigmoid() )
    def forward(self, x): return self.conv_stack(x)

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e, self.e_dim, self.beta = n_e, e_dim, beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1./self.n_e, 1./self.n_e)

    def forward(self, z):
        z_p = z.permute(0,2,3,1).contiguous()
        z_f = z_p.view(-1, self.e_dim)
        dist = torch.sum(z_f**2,1,True) + torch.sum(self.embedding.weight**2,1) - 2*torch.matmul(z_f, self.embedding.weight.t())
        min_idx = torch.argmin(dist, 1).unsqueeze(1)
        min_enc = torch.zeros(min_idx.shape[0], self.n_e, device=z.device).scatter_(1, min_idx, 1)
        z_q_embed = torch.matmul(min_enc, self.embedding.weight).view(z_p.shape)
        loss = torch.mean((z_q_embed.detach()-z_p)**2) + self.beta * torch.mean((z_q_embed - z_p.detach())**2)
        z_q_st = z_p + (z_q_embed - z_p).detach() # STE (Straight-Through Estimator)
        z_q_final = z_q_st.permute(0,3,1,2).contiguous() # This is z_q used for reconstruction
        e_mean = torch.mean(min_enc,0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean+1e-10)))
        return loss, z_q_final, perplexity, min_enc, min_idx # min_idx are the codebook indices

class VQVAE(nn.Module):
    def __init__(self, in_channels, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta, out_channels=1):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, h_dim, n_res_layers, res_h_dim)
        self.pre_quant_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vq = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim, out_channels=out_channels)

    def forward(self, x):
        z_e = self.encoder(x)
        z_e_pre_quant = self.pre_quant_conv(z_e) # Tensor before quantization
        vq_loss, z_q, perplexity, _, min_encoding_indices = self.vq(z_e_pre_quant) # z_q is quantized, min_encoding_indices are the indices
        x_recon = self.decoder(z_q)
        return vq_loss, x_recon, perplexity, z_q, z_e_pre_quant, min_encoding_indices

# --- Mahalanobis Scorer ---
class MahalanobisScorer:
    def __init__(self):
        self.mean_embedding = None
        self.inv_covariance = None
        self.fitted = False

    def fit_params(self, mean_embedding, inv_covariance):
        """Load pre-computed parameters."""
        self.mean_embedding = mean_embedding
        self.inv_covariance = inv_covariance
        if self.mean_embedding is not None and self.inv_covariance is not None:
            self.fitted = True
            print("Mahalanobis scorer parameters loaded and ready.")
        else:
            self.fitted = False
            print("Error: Mahalanobis mean embedding or inverse covariance is None during parameter loading.")

    def score(self, embeddings_np: np.ndarray) -> np.ndarray:
        if not self.fitted:
            print("Warning: Mahalanobis scorer not fitted. Returning NaN scores.")
            return np.full(embeddings_np.shape[0] if embeddings_np.ndim > 1 else 1, np.nan)
        
        if embeddings_np.ndim == 1: 
            embeddings_np = embeddings_np.reshape(1, -1)
        
        distances = []
        for i in range(embeddings_np.shape[0]):
            try:
                dist = mahalanobis(embeddings_np[i], self.mean_embedding, self.inv_covariance)
                distances.append(dist)
            except Exception as e:
                print(f"Error calculating Mahalanobis distance for a sample: {e}. Appending NaN.")
                distances.append(np.nan)
        return np.array(distances)

# --- Model Hyperparameters & Constants (YOU MUST REPLACE THESE) ---
# These should match your actual trained models and evaluation results

# medical_anomaly_detector/vqvae_models.py
# ... (VQVAE classes, MahalanobisScorer remain the same) ...

# --- Model Hyperparameters & Constants (MODIFIED) ---
# Thresholds will be loaded from JSON files by the pipeline now.
# norm_constants are still needed here as they are part of the hybrid score calculation logic.

LUNG_MODEL_CONFIG = {
    "params": {"in_channels": 1, "h_dim": 128, "res_h_dim": 64, "n_res_layers": 3,
               "n_embeddings": 128, "embedding_dim": 32, "beta": 0.25, "out_channels": 1},
    "img_size": 256,
    "patch_size_ratio": 16,
    # "hybrid_threshold": 0.55, # REMOVED - will be loaded from JSON
    "norm_constants": {"mse_min": 0.001, "mse_max": 0.1, "maha_min": 10.0, "maha_max": 100.0}
}

BREAST_MODEL_CONFIG = {
    "params": {"in_channels": 1, "h_dim": 128, "res_h_dim": 64, "n_res_layers": 3,
               "n_embeddings": 128, "embedding_dim": 32, "beta": 0.5, "out_channels": 1},
    "img_size": 512,
    "patch_size_ratio": 4,
    # "hybrid_threshold": 0.62, # REMOVED - will be loaded from JSON
    "norm_constants": {"mse_min": 0.002, "mse_max": 0.15, "maha_min": 15.0, "maha_max": 120.0}
}
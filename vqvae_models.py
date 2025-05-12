# medical_anomaly_detector/vqvae_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import mahalanobis

# --- VQ-VAE Model Components (Ensure these match your training script) ---
class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True), 
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True), 
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
        )
    def forward(self, x): 
        return x + self.res_block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)]
        )
    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )
    def forward(self, x):
        return self.conv_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_channels=1):
        super(Decoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.conv_stack(x)

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z_permute = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_permute.view(-1, self.e_dim)
        dist = (torch.sum(z_flat**2, dim=1, keepdim=True) +
                torch.sum(self.embedding.weight**2, dim=1) -
                2 * torch.matmul(z_flat, self.embedding.weight.t()))
        min_encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        z_q_embed = torch.matmul(min_encodings, self.embedding.weight).view(z_permute.shape)
        loss = torch.mean((z_q_embed.detach() - z_permute)**2) + \
               self.beta * torch.mean((z_q_embed - z_permute.detach())**2)
        z_q_st = z_permute + (z_q_embed - z_permute).detach() 
        z_q_final = z_q_st.permute(0, 3, 1, 2).contiguous()
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        return loss, z_q_final, perplexity, min_encodings, min_encoding_indices

class VQVAE(nn.Module):
    def __init__(self, in_channels, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta, out_channels=1):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_dim=in_channels, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim)
        self.pre_quant_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vq = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(in_dim=embedding_dim, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim, out_channels=out_channels)

    def forward(self, x):
        z_e = self.encoder(x)
        z_e_pre_quant = self.pre_quant_conv(z_e)
        vq_loss, z_q, perplexity, _, min_encoding_indices = self.vq(z_e_pre_quant)
        x_recon = self.decoder(z_q)
        return vq_loss, x_recon, perplexity, z_q, z_e_pre_quant, min_encoding_indices

class MahalanobisScorer:
    def __init__(self):
        self.mean_embedding = None
        self.inv_covariance = None
        self.fitted = False

    def fit_params(self, mean_embedding, inv_covariance):
        self.mean_embedding = mean_embedding
        self.inv_covariance = inv_covariance
        if self.mean_embedding is not None and self.inv_covariance is not None:
            self.fitted = True
            # print("Mahalanobis scorer parameters loaded and ready.") # Keep print for debugging if needed
        else:
            self.fitted = False
            print("Error: Mahalanobis mean or inv_covariance is None during param loading.")

    def score(self, embeddings_np: np.ndarray) -> np.ndarray:
        if not self.fitted:
            print("Warning: Mahalanobis scorer not fitted. Returning NaN scores.")
            return np.full(embeddings_np.shape[0] if embeddings_np.ndim > 1 else 1, np.nan)
        
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)
        
        if embeddings_np.shape[1] != self.mean_embedding.shape[0]:
             print(f"Error: Mahalanobis input embedding dim {embeddings_np.shape[1]} != "
                   f"scorer trained dim {self.mean_embedding.shape[0]}. Returning NaNs.")
             return np.full(embeddings_np.shape[0], np.nan)
        
        distances = []
        for i in range(embeddings_np.shape[0]):
            try:
                dist = mahalanobis(embeddings_np[i], self.mean_embedding, self.inv_covariance)
                distances.append(dist)
            except Exception as e:
                print(f"Error calculating Mahalanobis distance for sample {i}: {e}. Appending NaN.")
                distances.append(np.nan)
        return np.array(distances)

LUNG_MODEL_TYPE_ID = "lung" 
BREAST_MODEL_TYPE_ID = "breast"

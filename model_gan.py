import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.utils.spectral_norm as spectral_norm
from torchinfo import summary

def add_pool(x, nd_to_sample):
    if nd_to_sample is None:
        return x
    
    # Handle empty input
    if x.size(0) == 0:
        return torch.zeros(0, x.size(1), device=x.device)
    
    batch_size = int(torch.max(nd_to_sample).item()) + 1
    pooled_x = torch.zeros(batch_size, x.size(1), device=x.device)
    count = torch.zeros(batch_size, device=x.device)
    
    # Vectorized aggregation using index_add
    valid_indices = (nd_to_sample < batch_size)
    nd_to_sample = nd_to_sample[valid_indices]
    x = x[valid_indices]
    
    pooled_x.index_add_(0, nd_to_sample, x)
    count.index_add_(0, nd_to_sample, torch.ones_like(nd_to_sample, dtype=torch.float))
    
    # Avoid division by zero
    count = torch.clamp(count, min=1)
    return pooled_x / count.unsqueeze(1)

def compute_gradient_penalty(D, real, fake, given_y_pooled, nd_to_sample):
    device = real.device
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    
    d_interpolates = D(interpolates, given_y_pooled, nd_to_sample)
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=False):
    layers = []
    
    if upsample:
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=1, padding=p)
    else:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
    
    if spec_norm:
        layers.append(spectral_norm(conv))
    else:
        layers.append(conv)
        
    # Add activation
    if act == "leaky":
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    elif act == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif act == "tanh":
        layers.append(nn.Tanh())
        
    return nn.Sequential(*layers)

class CMP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(3*in_channels, 2*in_channels, 3, 1, 1, act="leaky"),
            conv_block(2*in_channels, in_channels, 3, 1, 1, act="leaky")
        )

    def forward(self, feats, edges):
        edges = edges.view(-1, 3)
        num_nodes = feats.size(0)
        device = feats.device
        
        # Initialize pooling tensors
        pooled_pos = torch.zeros_like(feats)
        pooled_neg = torch.zeros_like(feats)
        
        # Process positive edges (relationship type > 0)
        pos_mask = edges[:, 1] > 0
        if pos_mask.any():
            pos_edges = edges[pos_mask]
            sources = pos_edges[:, 0].long().clamp(0, num_nodes-1)
            targets = pos_edges[:, 2].long().clamp(0, num_nodes-1)
            pooled_pos.index_add_(0, targets, feats[sources])
        
        # Process negative edges (relationship type < 0)
        neg_mask = edges[:, 1] < 0
        if neg_mask.any():
            neg_edges = edges[neg_mask]
            sources = neg_edges[:, 0].long().clamp(0, num_nodes-1)
            targets = neg_edges[:, 2].long().clamp(0, num_nodes-1)
            pooled_neg.index_add_(0, targets, feats[sources])
        
        # Combine features and encode
        combined = torch.cat([feats, pooled_pos, pooled_neg], dim=1)
        return self.encoder(combined)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = 8
        self.latent_dim = 128
        self.num_types = 18
        
        # Initial projection
        self.l1 = nn.Linear(self.latent_dim + self.num_types, 16 * self.init_size ** 2)
        
        # Mask encoder
        self.enc_1 = nn.Sequential(
            conv_block(1, 32, 3, 2, 1, act="leaky"),  # 32x32 -> 16x16
            conv_block(32, 32, 3, 2, 1, act="leaky"),  # 16x16 -> 8x8
            conv_block(32, 16, 3, 1, 1, act="leaky")   # Maintain 8x8
        )
        
        # Graph processing
        self.cmp_layers = nn.ModuleList([CMP(16) for _ in range(4)])
        self.agg_layers = nn.ModuleList([
            conv_block(16, 16, 3, 1, 1, act="leaky") for _ in range(3)
        ])
        
        # Image decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            conv_block(16, 16, 3, 1, 1, act="leaky"),
            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            conv_block(16, 16, 3, 1, 1, act="leaky"),
            nn.Upsample(scale_factor=2),  # 32x32 -> 64x64
            conv_block(16, 1, 3, 1, 1, act="tanh")  # Final output
        )

    def forward(self, z, given_m, given_y, given_w, nd_to_sample):
        num_rooms = given_m.size(0)
        batch_size = int(torch.max(nd_to_sample).item()) + 1

        # Encode room masks
        m = self.enc_1(given_m)  # [num_rooms, 16, 8, 8]

        # Prepare latent vectors
        z_expanded = z[nd_to_sample]  # Map to rooms
        z_y = torch.cat([z_expanded, given_y], dim=1)
        init_feat = self.l1(z_y).view(num_rooms, 16, 8, 8)
        
        # Combine mask features and latent code
        x = m + init_feat
        x = F.dropout(x, p=0.2, training=self.training)

        # Process through CMP layers
        for cmp_layer in self.cmp_layers:
            x = cmp_layer(x, given_w)
                    
        # Aggregate room features per sample
        pooled_x = []
        for i in range(batch_size):
            mask = (nd_to_sample == i)
            if mask.any():
                pooled_x.append(torch.mean(x[mask], dim=0, keepdim=True))
            else:
                pooled_x.append(torch.zeros(1, 16, 8, 8, device=x.device))
        pooled_x = torch.cat(pooled_x, dim=0)
        
        # Refine aggregated features
        for agg_layer in self.agg_layers:
            pooled_x = agg_layer(pooled_x)
        
        # Generate final image
        return self.decoder(pooled_x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_types = 18
        
        # Type embedding network
        self.type_encoder = nn.Sequential(
            spectral_norm(nn.Linear(self.num_types, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 4 * 64 * 64))
        )
        
        # Image processing
        self.encoder = nn.Sequential(
            conv_block(5, 32, 4, 2, 1, act="leaky", spec_norm=True),  # 64x64 -> 32x32
            conv_block(32, 64, 4, 2, 1, act="leaky", spec_norm=True), # 32x32 -> 16x16
            conv_block(64, 128, 4, 2, 1, act="leaky", spec_norm=True),# 16x16 -> 8x8
            conv_block(128, 256, 4, 2, 1, act="leaky", spec_norm=True) # 8x8 -> 4x4
        )
        
        # Final classification
        self.fc = spectral_norm(nn.Linear(256, 1))

    def forward(self, x, given_y, nd_to_sample=None):
        # Aggregate room types if needed
        if nd_to_sample is not None and given_y.size(0) != x.size(0):
            given_y = add_pool(given_y, nd_to_sample)
        
        # Encode room types
        y_embed = self.type_encoder(given_y)
        y_embed = y_embed.view(given_y.size(0), 4, 64, 64)
        
        # Combine with input image
        x = torch.cat([x, y_embed], dim=1)  # [batch_size, 5, 64, 64]
        
        # Process through encoder
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global pooling
        x = torch.flatten(x, 1)
        output = self.fc(x)
        
        # Soft clamping pour stabiliser l'entraînement
        return torch.tanh(output) * 20.0  # Clamp entre -20 et 20

def main():
    batch_size = 4
    z_dim = 128
    num_types = 18
    
    # Sample data
    z = torch.randn(batch_size, z_dim)
    n_rooms = 56  # 14+12+18+12
    given_m = torch.randn(n_rooms, 1, 32, 32)
    given_y = torch.zeros(n_rooms, num_types)
    given_w = torch.randint(0, n_rooms, (376, 3))
    
    # Node to sample mapping
    nd_to_sample = torch.cat([
        torch.zeros(14), 
        torch.ones(12), 
        torch.full((18,), 2), 
        torch.full((12,), 3)
    ]).long()
    
    # Initialize models
    G = Generator()
    D = Discriminator()
    
    # Print summaries
    print("\n" + "="*50)
    print("GENERATOR SUMMARY")
    print("="*50)
    summary(G, input_data=[z, given_m, given_y, given_w, nd_to_sample], depth=5)
    
    print("\n" + "="*50)
    print("DISCRIMINATOR SUMMARY")
    print("="*50)
    summary(D, input_data=[
        torch.randn(batch_size, 1, 64, 64),
        given_y,
        nd_to_sample
    ], depth=5)
    
    # Test forward passes
    print("\n" + "="*50)
    print("MODEL OUTPUT SHAPES")
    print("="*50)
    with torch.no_grad():
        fake = G(z, given_m, given_y, given_w, nd_to_sample)
        print(f"Generator output shape: {fake.shape}")  # [4, 1, 64, 64]
        
        validity = D(fake, given_y, nd_to_sample)
        print(f"Discriminator output shape: {validity.shape}")  # [4, 1]
        print(f"Discriminator output range: min={validity.min().item():.4f}, max={validity.max().item():.4f}")

if __name__ == "__main__":
    main()
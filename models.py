import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.utils.spectral_norm as spectral_norm
from torchinfo import summary

def add_pool(x, nd_to_sample):
    if nd_to_sample is None:
        return x
    
    if x.size(0) == 0:
        return torch.zeros(0, x.size(1), device=x.device)
    
    batch_size = int(torch.max(nd_to_sample).item()) + 1
    if batch_size == 0:
        return torch.zeros(1, x.size(1), device=x.device)
    pooled_x = torch.zeros(batch_size, x.size(1), device=x.device)
    count = torch.zeros(batch_size, device=x.device)
    
    valid_indices = (nd_to_sample < batch_size)
    nd_to_sample = nd_to_sample[valid_indices]
    x = x[valid_indices]
    
    pooled_x.index_add_(0, nd_to_sample, x)
    count.index_add_(0, nd_to_sample, torch.ones_like(nd_to_sample, dtype=torch.float))
    
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

def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=False, bn=False):
    layers = []
    
    if upsample:
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
    
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
    
    if spec_norm:
        layers.append(spectral_norm(conv))
    else:
        layers.append(conv)
    
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
        
    if act == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif act == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif act == "tanh":
        layers.append(nn.Tanh())
        
    return nn.Sequential(*layers)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class CMP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(3*in_channels, 2*in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(2*in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, feats, edges):
        edges = edges.view(-1, 3)
        num_nodes = feats.size(0)
        device = feats.device
        
        pooled_pos = torch.zeros_like(feats)
        pooled_neg = torch.zeros_like(feats)
        
        all_sources = edges[:, 0].long().clamp(0, num_nodes-1)
        all_targets = edges[:, 2].long().clamp(0, num_nodes-1)
        edge_types = edges[:, 1]
        
        pos_mask = (edge_types > 0)
        neg_mask = (edge_types < 0)
        
        pooled_pos.index_add_(
            0, 
            all_targets[pos_mask], 
            feats[all_sources[pos_mask]]
        )
        pooled_neg.index_add_(
            0, 
            all_targets[neg_mask], 
            feats[all_sources[neg_mask]]
        )
        
        combined = torch.cat([feats, pooled_pos, pooled_neg], dim=1)
        return feats + self.encoder(combined)

class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = spectral_norm(nn.Linear(in_channels, in_channels))
        self.key = spectral_norm(nn.Linear(in_channels, in_channels))
        self.value = spectral_norm(nn.Linear(in_channels, in_channels))
        self.scale = in_channels ** -0.5
        
    def forward(self, x):
        batch_size, num_rooms, channels, h, w = x.size()
        x_pooled = x.view(batch_size * num_rooms, channels, -1).mean(dim=-1)
        x_pooled = x_pooled.view(batch_size, num_rooms, channels)
        
        q = self.query(x_pooled)
        k = self.key(x_pooled)
        v = self.value(x_pooled)
        
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        aggregated = torch.bmm(attn, v)
        aggregated = aggregated.mean(dim=1)
        return aggregated.view(batch_size, channels, 1, 1)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = 8
        self.latent_dim = 128
        self.num_types = 19
        
        self.l1 = spectral_norm(nn.Linear(128 + 19, 32 * 8 * 8))
        
        self.enc_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 32, 3, 1, 1)),
            nn.LeakyReLU(0.2),
        )
        
        self.cmp_layers = nn.ModuleList([CMP(32) for _ in range(3)])
        self.attention_pool = AttentionPooling(32)
        
        # Décodeur mis à jour
        self.decoder = nn.Sequential(
            UpsampleBlock(32, 64),   # 8x8 -> 16x16
            UpsampleBlock(64, 128),  # 16x16 -> 32x32
            spectral_norm(nn.Conv2d(128, 64, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 32, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(32, 1, 3, padding=1)),
            nn.Sigmoid()
        )

    def forward(self, z, given_m, given_y, given_w, nd_to_sample):
        num_rooms = given_m.size(0)
        batch_size = torch.max(nd_to_sample).item() + 1

        m = self.enc_1(given_m)
        
        z_expanded = z[nd_to_sample]
        z_y = torch.cat([z_expanded, given_y], dim=1)
        init_feat = self.l1(z_y).view(num_rooms, 32, 8, 8)
        
        if m.size(2) != 8 or m.size(3) != 8:
            m = F.interpolate(m, size=(8, 8), mode='bilinear')
        
        x = m + init_feat
        x = F.dropout(x, p=0.5, training=self.training)

        for cmp_layer in self.cmp_layers:
            x = cmp_layer(x, given_w)
        
        pooled_x = []
        for i in range(batch_size):
            mask = (nd_to_sample == i)
            if mask.any():
                sample_features = x[mask].unsqueeze(0)
                pooled = self.attention_pool(sample_features)
                pooled_x.append(pooled)
            else:
                pooled_x.append(torch.zeros(1, 32, 1, 1, device=x.device))
                
        pooled_x = torch.cat(pooled_x, dim=0)
        
        # Ajout de l'interpolation pour obtenir 8x8
        pooled_x = F.interpolate(pooled_x, size=(8, 8), mode='nearest')
        return self.decoder(pooled_x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_types = 19
        
        self.type_encoder = nn.Sequential(
            spectral_norm(nn.Linear(19, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 1 * 32 * 32))
        )
        
        self.encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 32, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(32, 64, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Changement: 256 entrées -> 16 sorties
        self.fc = spectral_norm(nn.Linear(256, 16))

    def forward(self, x, given_y, nd_to_sample=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Conversion one-hot si nécessaire
        if given_y.size(1) != self.num_types:
            given_y = F.one_hot(torch.argmax(given_y, dim=1), 
                               num_classes=self.num_types).float()
            
        
        # Encodage direct sans pooling supplémentaire
        y_embed = self.type_encoder(given_y)
        y_embed = y_embed.view(-1, 1, 32, 32)
        
        # Ajustement de y_embed_expanded selon les cas
        if nd_to_sample is not None:
            # Cas 1 : même nombre de samples dans x et nd_to_sample => on suppose que x contient des masques par pièce
            if y_embed.shape[0] == nd_to_sample.shape[0]:
                y_embed_expanded = y_embed
            # Cas 2 : y_embed correspond à un vecteur par batch, on doit étendre
            elif y_embed.shape[0] == nd_to_sample.max().item() + 1:
                y_embed_expanded = y_embed[nd_to_sample]
            else:
                raise ValueError(
                    f"Incohérence détectée : x={x.shape[0]}, y_embed={y_embed.shape[0]}, "
                    f"nd_to_sample={nd_to_sample.shape}"
                )
        else:
            # Cas sans nd_to_sample → batch direct
            if y_embed.shape[0] == x.shape[0]:
                y_embed_expanded = y_embed
            else:
                raise ValueError(
                    f"Incohérence détectée : x={x.shape[0]}, y_embed={y_embed.shape[0]}, nd_to_sample=None"
                )




        # y_embed_expanded = F.interpolate(y_embed_expanded, size=x.shape[2:], mode='nearest')

        # print("x:", x.shape)
        # print("y_embed_expanded:", y_embed_expanded.shape)
        # print("nd_to_sample:", nd_to_sample.shape, nd_to_sample.max())

        if y_embed_expanded.shape[2:] != x.shape[2:]:
            y_embed_expanded = F.interpolate(y_embed_expanded, size=x.shape[2:], mode='bilinear', align_corners=False)

        print(f"x.shape: {x.shape}, y_embed_expanded.shape: {y_embed_expanded.shape}")
        
        x = torch.cat([x, y_embed_expanded], dim=1)
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        if nd_to_sample is not None:
            features = add_pool(features, nd_to_sample)
        
        # Modification finale: 16 valeurs par batch -> aplaties en [batch_size*16, 1]
        out = self.fc(features)
        out = out.view(-1, 1)  # Forme finale [64, 1] pour batch_size=4
        return out

    

# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.init_size = 8
#         self.latent_dim = 128
#         self.num_types = 18
        
#         # Réduction des canaux
#         self.l1 = spectral_norm(nn.Linear(self.latent_dim + self.num_types, 32 * self.init_size ** 2))
        
#         self.enc_1 = nn.Sequential(
#             spectral_norm(nn.Conv2d(1, 64, 3, 2, 1)),
#             nn.LeakyReLU(0.2),
#             spectral_norm(nn.Conv2d(64, 32, 3, 2, 1)),
#             nn.LeakyReLU(0.2),
#         )
        
#         # Réduction des canaux dans CMP
#         self.cmp_layers = nn.ModuleList([CMP(32) for _ in range(3)])
#         self.attention_pool = AttentionPooling(32)
        
#         self.decoder = nn.Sequential(
#             UpsampleBlock(32, 64),
#             UpsampleBlock(64, 32),
#             UpsampleBlock(32, 16),
#             spectral_norm(nn.Conv2d(16, 8, 3, 1, 1)),
#             nn.LeakyReLU(0.2),
#             spectral_norm(nn.Conv2d(8, 1, 3, padding=1)),
#             nn.Tanh()
#         )

#     def forward(self, z, given_m, given_y, given_w, nd_to_sample):
#         num_rooms = given_m.size(0)
#         batch_size = int(torch.max(nd_to_sample).item()) + 1

#         m = self.enc_1(given_m)
#         z_expanded = z[nd_to_sample]
#         z_y = torch.cat([z_expanded, given_y], dim=1)
#         init_feat = self.l1(z_y).view(num_rooms, 32, self.init_size, self.init_size)  # 64 -> 32
        
#         x = m + init_feat
#         x = F.dropout(x, p=0.5, training=self.training)

#         for cmp_layer in self.cmp_layers:
#             x = cmp_layer(x, given_w)
        
#         pooled_x = []
#         for i in range(batch_size):
#             mask = (nd_to_sample == i)
#             if mask.any():
#                 sample_features = x[mask].unsqueeze(0)
#                 pooled = self.attention_pool(sample_features)
#                 pooled_x.append(pooled)
#             else:
#                 pooled_x.append(torch.zeros(1, 32, 1, 1, device=x.device))  # 64->32
                
#         pooled_x = torch.cat(pooled_x, dim=0)
#         pooled_x = F.interpolate(pooled_x, size=(8, 8), mode='nearest')
#         return self.decoder(pooled_x)

    
# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.num_types = 18
        
#         self.type_encoder = nn.Sequential(
#             spectral_norm(nn.Linear(self.num_types, 64)),
#             nn.LeakyReLU(0.2),
#             spectral_norm(nn.Linear(64, 128)),
#             nn.LeakyReLU(0.2),
#             spectral_norm(nn.Linear(128, 1 * 32 * 32))  # 32x32 au lieu de 64x64
#         )
        
#         # Architecture adaptée pour 32x32
#         self.encoder = nn.Sequential(
#             spectral_norm(nn.Conv2d(2, 32, 4, 2, 1)),  # [n, 32, 16, 16]
#             nn.LeakyReLU(0.2),
#             spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)),  # [n, 64, 8, 8]
#             nn.LeakyReLU(0.2),
#             spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), # [n, 128, 4, 4]
#             nn.LeakyReLU(0.2),
#             spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), # [n, 256, 2, 2]
#             nn.LeakyReLU(0.2),
#             nn.AdaptiveAvgPool2d(1)  # [n, 256, 1, 1]
#         )
        
#         self.fc = spectral_norm(nn.Linear(256, 1))

#     def forward(self, x, given_y, nd_to_sample=None):
#         """
#         x: masques de pièces [total_rooms, 1, 32, 32]
#         given_y: types de pièces [total_rooms, 18]
#         nd_to_sample: mapping pièce→batch [total_rooms]
#         """
#         # Étape 1: Pooling des types par batch
#         if nd_to_sample is not None:
#             # Cas 1: Pooling si nécessaire
#             if given_y.size(0) != nd_to_sample.max().item() + 1:
#                 given_y = add_pool(given_y, nd_to_sample)  # [batch, 18]
        
#         # Étape 2: Encodage des types
#         y_embed = self.type_encoder(given_y)  # [batch, 32*32]
#         y_embed = y_embed.view(-1, 1, 32, 32)  # [batch, 1, 32, 32]
        
#         # Étape 3: Dupliquer pour chaque pièce du batch
#         if nd_to_sample is not None:
#             y_embed_expanded = y_embed[nd_to_sample]  # [total_rooms, 1, 32, 32]
#         else:
#             y_embed_expanded = y_embed  # Cas sans batch (single sample)
        
#         # Étape 4: Concaténation avec les masques
#         x = torch.cat([x, y_embed_expanded], dim=1)  # [total_rooms, 2, 32, 32]
        
#         # Étape 5: Traitement CNN
#         features = self.encoder(x)  # [total_rooms, 256, 1, 1]
#         features = features.view(features.size(0), -1)  # [total_rooms, 256]
        
#         # Étape 6: Pooling par batch (si nécessaire)
#         if nd_to_sample is not None:
#             features = add_pool(features, nd_to_sample)  # [batch, 256]
        
#         return self.fc(features)  # [batch, 1] ou [total_rooms, 1]

def main():
    batch_size = 4
    z_dim = 128
    num_types = 19
    
    z = torch.randn(batch_size, z_dim)
    n_rooms = 56
    given_m = torch.randn(n_rooms, 1, 32, 32)
    given_y = torch.zeros(n_rooms, num_types)
    given_w = torch.randint(0, n_rooms, (376, 3))
    
    nd_to_sample = torch.cat([
        torch.zeros(14), 
        torch.ones(12), 
        torch.full((18,), 2), 
        torch.full((12,), 3)
    ]).long()
    
    G = Generator()
    D = Discriminator()
    
    print("\n" + "="*50)
    print("GENERATOR SUMMARY")
    print("="*50)
    summary(G, input_data=[z, given_m, given_y, given_w, nd_to_sample], depth=5)
    
    print("\n" + "="*50)
    print("DISCRIMINATOR SUMMARY")
    print("="*50)
    summary(D, input_data=[
        given_m,
        given_y,
        nd_to_sample
    ], depth=5)

    # Test forward passes
    print("\n" + "="*50)
    print("MODEL OUTPUT SHAPES")
    print("="*50)
    with torch.no_grad():
        fake = G(z, given_m, given_y, given_w, nd_to_sample)
        print(f"Generator output shape: {fake.shape}")
        
        validity = D(given_m, given_y, nd_to_sample)
        print(f"Discriminator output shape: {validity.shape}")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn.utils.spectral_norm as spectral_norm
from torchinfo import summary


def add_pool(x, nd_to_sample, batch_size=None):
    """Agrège les caractéristiques par échantillon avec taille de batch spécifiable"""
    if nd_to_sample is None or x.size(0) == 0:
        return x
    
    if nd_to_sample.numel() == 0:
        return torch.zeros(0, x.size(1), device=x.device)
    
    if len(nd_to_sample) != x.size(0):
        raise RuntimeError(
            f"Dimension mismatch: nd_to_sample ({len(nd_to_sample)}) "
            f"must match first dimension of x ({x.size(0)})"
        )
    
    # Calcul de la taille du batch si non spécifiée
    if batch_size is None:
        max_index = torch.max(nd_to_sample).item()
        batch_size = max_index + 1 if max_index >= 0 else 0
    
    # Cas spécial: aucun échantillon valide
    if batch_size == 0:
        return torch.zeros(0, x.size(1), device=x.device)
    
    # Flatten des caractéristiques
    if x.dim() != 2:
        x = x.view(x.size(0), -1)
    
    pooled_x = torch.zeros(batch_size, x.size(1), device=x.device)
    count = torch.zeros(batch_size, device=x.device)
    
    # Filtrer les indices valides
    valid_indices = (nd_to_sample >= 0) & (nd_to_sample < batch_size)
    
    if not valid_indices.any():
        return pooled_x
    
    nd_to_sample_valid = nd_to_sample[valid_indices]
    x_valid = x[valid_indices]
    
    pooled_x.index_add_(0, nd_to_sample_valid, x_valid)
    count.index_add_(0, nd_to_sample_valid, torch.ones_like(nd_to_sample_valid, dtype=torch.float))
    
    count = torch.clamp(count, min=1)
    return pooled_x / count.unsqueeze(1)

def compute_gradient_penalty(D, real_samples, fake_samples, given_y, nd_to_sample):
    """Version corrigée avec gestion des arguments conditionnels"""
    device = real_samples.device
    batch_size = real_samples.size(0)  # Correction: utiliser batch_size réel
    
    # Création de l'interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # Forward pass - gestion du conditionnement
    if given_y is not None:
        # Si given_y est fourni, on le tronque si nécessaire
        if given_y.size(0) != batch_size:
            given_y = given_y[:batch_size] 
        d_interpolates = D(
            interpolates, 
            given_y=given_y, 
            nd_to_sample=None  # Pass None since given_y is already aggregated
        )
    else:
        # Appel sans conditionnement
        d_interpolates = D(
            interpolates, 
            given_y=None, 
            nd_to_sample=None
        )
    
    # Calcul des gradients
    grad_outputs = torch.ones_like(d_interpolates)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True 
    )[0]
    
    # Reshape et calcul de la pénalité
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# === BLOCS ===

class UpsampleBlock(nn.Module):
    """Bloc d'upsampling avec convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.InstanceNorm2d(out_channels),  # Correction: BatchNorm → InstanceNorm
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class CMP(nn.Module):
    """Conditional Message Passing pour graphes"""
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(3 * in_channels, 2 * in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(2 * in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, feats, edges):
        edges = edges.view(-1, 3)
        num_nodes = feats.size(0)
        device = feats.device
        
        # Filtrer les arêtes valides - CORRECTION AJOUTÉE
        valid_mask = (edges[:, 0] >= 0) & (edges[:, 0] < num_nodes) & \
                     (edges[:, 2] >= 0) & (edges[:, 2] < num_nodes)
        edges = edges[valid_mask]
        
        # Initialisation des caractéristiques agrégées
        pooled_pos = torch.zeros_like(feats)
        pooled_neg = torch.zeros_like(feats)
        
        # Extraction des connexions
        all_sources = edges[:, 0].long()
        all_targets = edges[:, 2].long()
        edge_types = edges[:, 1]
        
        # Masques pour relations positives/négatives
        pos_mask = (edge_types > 0)
        neg_mask = (edge_types < 0)
        
        # Agrégation des messages
        pooled_pos.index_add_(0, all_targets[pos_mask], feats[all_sources[pos_mask]])
        pooled_neg.index_add_(0, all_targets[neg_mask], feats[all_sources[neg_mask]])
        
        # Combinaison et transformation
        combined = torch.cat([feats, pooled_pos, pooled_neg], dim=1)
        return feats + self.encoder(combined)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = 8
        self.latent_dim = 128
        self.num_types = 18
        
        self.l1 = spectral_norm(nn.Linear(128 + 18, 32 * 8 * 8))
        
        self.enc_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 32, 3, 1, 1)),
            nn.LeakyReLU(0.2),
        )
        
        # Correction: ajout convolution d'ajustement
        self.adjust_channel = spectral_norm(nn.Conv2d(32, 32, 1))
        
        self.cmp_layers = nn.ModuleList([CMP(32) for _ in range(3)])
        
        # Décodeur pour masques individuels
        self.decoder = nn.Sequential(
            UpsampleBlock(32, 64),
            UpsampleBlock(64, 128),
            spectral_norm(nn.Conv2d(128, 64, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 32, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(32, 1, 3, padding=1)),
            nn.Sigmoid()
        )
        
        # Initialisation des poids
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z, given_m, given_y, given_w, nd_to_sample):
        num_rooms = given_m.size(0)

        if given_m.dim() != 4 or given_m.size(1) != 1:
            given_m = given_m.unsqueeze(1)
        
        # Vérification de dimension - CORRECTION AJOUTÉE
        assert given_m.size(0) == given_y.size(0) == len(nd_to_sample), \
            "Mismatch in number of rooms"

        # Encodage des masques
        m = self.enc_1(given_m)
        
        # Ajustement des canaux si nécessaire
        if m.size(1) != 32:
            m = self.adjust_channel(m)
        if m.size(2) != 8 or m.size(3) != 8:
            m = F.interpolate(m, size=(8, 8), mode='bilinear')
        
        # Fusion bruit + type
        z_expanded = z[nd_to_sample]
        z_y = torch.cat([z_expanded, given_y], dim=1)
        init_feat = self.l1(z_y).view(num_rooms, 32, 8, 8)
        
        x = m + init_feat
        x = F.dropout(x, p=0.5, training=self.training)

        # Traitement relationnel
        for cmp_layer in self.cmp_layers:
            x = cmp_layer(x, given_w)

        x = x + 0.01 * torch.randn_like(x)
        
        # Décoder chaque pièce individuellement
        return self.decoder(x)  # [num_rooms, 1, 32, 32]
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_types = 18
        
        self.type_encoder = nn.Sequential(
            spectral_norm(nn.Linear(18, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 256))
        )
        
        # Définition séparée des couches pour permettre l'extraction de features
        self.conv1 = spectral_norm(nn.Conv2d(257, 32, 3, 1, 1))
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = spectral_norm(nn.Conv2d(32, 64, 3, 2, 1))
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = spectral_norm(nn.Conv2d(64, 128, 3, 2, 1))
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, 3, 2, 1))
        self.lrelu4 = nn.LeakyReLU(0.2)
        self.conv5 = spectral_norm(nn.Conv2d(256, 256, 3, 2, 1))
        self.lrelu5 = nn.LeakyReLU(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = spectral_norm(nn.Linear(256, 1))
        
        # Couche à utiliser pour l'extraction de features (après conv4)
        self.feature_layer_index = 7  # Index après la 4ème convolution
        
        # Initialisation des poids
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _prepare_input(self, x, given_y=None, nd_to_sample=None):
        """Préparation commune des entrées pour forward et feature_extractor"""
        # Gestion des dimensions
        if x.dim() == 5:
            x = x.squeeze(2)
        elif x.dim() != 4:
            raise ValueError(f"Invalid input dimensions: {x.dim()}")
        
        # FORCEZ la taille à 32x32
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        
        batch_size = x.size(0)
        
        # Si nd_to_sample n'est pas fourni
        if nd_to_sample is None:
            nd_to_sample = torch.arange(batch_size, device=x.device)
        
        # Gestion des labels manquants
        if given_y is None:
            given_y = torch.zeros(nd_to_sample.size(0), self.num_types, device=x.device)
        elif given_y.dim() == 1:
            given_y = F.one_hot(given_y.long(), num_classes=self.num_types).float()
        
        # Vérification de dimension
        assert given_y.size(0) == len(nd_to_sample), \
            "Mismatch in number of room labels"
        
        # Encodage des types
        y_embed_room = self.type_encoder(given_y)
        y_embed = add_pool(y_embed_room, nd_to_sample, batch_size=batch_size)
        y_embed = y_embed[:, :256]  # Conserver seulement 256 features
        
        # Préparation spatiale
        y_embed = y_embed.view(batch_size, 256, 1, 1)
        y_embed = F.interpolate(
            y_embed, 
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Concaténation finale
        return torch.cat([x, y_embed], dim=1)

    def feature_extractor(self, x, given_y=None, nd_to_sample=None):
        """Extrait les caractéristiques intermédiaires du discriminateur"""
        x_combined = self._prepare_input(x, given_y, nd_to_sample)
        
        # Passage à travers les couches jusqu'au point d'extraction
        x = self.conv1(x_combined)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        features = self.lrelu4(x)  # Point d'extraction des features
        
        return features

    def forward(self, x, given_y=None, nd_to_sample=None):
        """Passe forward complète avec sortie de validité"""
        x_combined = self._prepare_input(x, given_y, nd_to_sample)
        
        # Passage à travers toutes les couches
        x = self.conv1(x_combined)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x = self.lrelu4(x)
        x = self.conv5(x)
        x = self.lrelu5(x)
        x = self.avgpool(x)
        
        # Couche finale
        x = x.view(x.size(0), -1)
        return self.fc(x)

# === TEST ===
def main():
    # Configuration des dimensions
    batch_size = 4
    z_dim = 128
    num_types = 18
    total_rooms = 465
    total_edges = 1943
    
    # Données simulées
    z = torch.randn(batch_size, z_dim)
    given_m = torch.randn(total_rooms, 1, 32, 32)
    
    # One-hot valide pour générateur (18 classes)
    given_y_gen = torch.zeros(total_rooms, num_types)
    given_y_gen[torch.arange(total_rooms), torch.randint(0, num_types, (total_rooms,))] = 1
    
    # One-hot valide pour discriminateur (18 classes)
    given_y_disc = torch.zeros(total_rooms, 18)
    given_y_disc[torch.arange(total_rooms), torch.randint(0, 18, (total_rooms,))] = 1
    
    given_w = torch.randint(0, total_rooms, (total_edges, 3))
    
    # Mapping pièce -> échantillon
    room_counts = [120, 95, 130, 120]
    nd_to_sample = torch.cat([
        torch.full((count,), idx) for idx, count in enumerate(room_counts)
    ]).long()
    
    # ================== VÉRIFICATIONS AVANT L'APPEL ==================
    print("\n" + "="*50)
    print("DIMENSION CHECKS BEFORE MODEL CALL")
    print("="*50)
    print(f"z shape: {z.shape}")  # [4, 128]
    print(f"given_y_gen shape: {given_y_gen.shape}")  # [465, 18]
    print(f"nd_to_sample shape: {nd_to_sample.shape}")  # [465]
    
    # Vérification de l'expansion de z
    z_expanded = z[nd_to_sample]
    print(f"z_expanded shape: {z_expanded.shape}")  # [465, 128]
    
    # Vérification de la concaténation
    z_y = torch.cat([z_expanded, given_y_gen], dim=1)
    print(f"z_y shape: {z_y.shape}")  # [465, 146]
    
    # Initialisation des modèles
    G = Generator()
    D = Discriminator()
    
    # ================== RÉSUMÉS DES MODÈLES ==================
    print("\n" + "="*50)
    print("GENERATOR SUMMARY")
    print("="*50)
    summary(G, input_data=[z, given_m, given_y_gen, given_w, nd_to_sample], depth=5)
    
    print("\n" + "="*50)
    print("DISCRIMINATOR SUMMARY")
    print("="*50)
    # Créer un batch d'images globales (1 par étage) pour le résumé du Discriminateur
    # Chaque image globale = composition des pièces d'un étage
    # Simuler 4 images globales (batch_size=4) de taille [4, 1, 256, 256]
    global_image = torch.randn(batch_size, 1, 256, 256)
    # Créer un nd_to_sample adapté pour les images globales (1 par échantillon)
    global_nd_to_sample = torch.arange(batch_size).long()
    # Labels agrégés au niveau étage (batch_size, 18)
    given_y_global = torch.zeros(batch_size, 18)
    given_y_global[torch.arange(batch_size), torch.randint(0, 18, (batch_size,))] = 1
    
    summary(D, input_data=[global_image, given_y_global, global_nd_to_sample], depth=5)

    # ================== TEST DES PASSES FORWARD ==================
    print("\n" + "="*50)
    print("MODEL OUTPUT SHAPES")
    print("="*50)
    with torch.no_grad():
        # Génération des masques de pièces
        fake_rooms = G(z, given_m, given_y_gen, given_w, nd_to_sample)
        print(f"Generator output shape: {fake_rooms.shape}")  # [465, 1, 32, 32]
        
        # Composition des pièces en images globales (batch_size, 1, 256, 256)
        # Ici on simule avec des randn pour le test
        fake_global = torch.randn(batch_size, 1, 256, 256)
        
        # Évaluation des images globales générées
        validity = D(fake_global, given_y_global, global_nd_to_sample)
        print(f"Discriminator output shape: {validity.shape}")  # [4, 1]

if __name__ == "__main__":
    main()
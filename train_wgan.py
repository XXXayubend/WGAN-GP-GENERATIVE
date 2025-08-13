import argparse
import os
import numpy as np
import csv
from floorplan import FloorplanImageDataset, floorplan_collate_fn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from model import Discriminator, Generator, compute_gradient_penalty, add_pool

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")  
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches") 
parser.add_argument("--d_lr", type=float, default=1e-6, help="adam: learning rate")
parser.add_argument("--g_lr", type=float, default=1e-6, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.0, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling (epochs)")
parser.add_argument("--exmps_folder", type=str, default='exmps', help="destination folder")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--target_set", type=int, default=8, choices=[5, 6, 7, 8], help="which split to remove")
parser.add_argument("--data_path", type=str, default='data/sample_list.txt', help="path to the dataset")
parser.add_argument("--lambda_gp", type=int, default=10, help="lambda for gradient penalty")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
opt = parser.parse_args()

def compose_global_image(room_masks, nd_to_sample, batch_size, target_size=32):
    # Vérification des dimensions
    if room_masks.dim() == 3:
        room_masks = room_masks.unsqueeze(1)
    elif room_masks.dim() != 4:
        raise ValueError(f"room_masks must be 3D or 4D, got {room_masks.dim()}D")

    # FORCEZ la taille à 32x32
    if room_masks.size(2) != 32 or room_masks.size(3) != 32:
        room_masks = F.interpolate(
            room_masks, 
            size=(32, 32), 
            mode='bilinear', 
            align_corners=False
        )
        print(f"Resized masks to {room_masks.shape}")

    if room_masks.size(2) != target_size:
        room_masks = F.interpolate(room_masks, size=(target_size, target_size))

    C, H, W = room_masks.shape[1], room_masks.shape[2], room_masks.shape[3]
    device = room_masks.device
    global_images = torch.zeros(batch_size, C, H, W, device=device)
    
    # Vérification de cohérence
    if len(nd_to_sample) != room_masks.size(0):
        raise RuntimeError(
            f"Dimension mismatch: nd_to_sample ({len(nd_to_sample)}) "
            f"and room_masks ({room_masks.size(0)}) must have same length"
        )
    
    # Agrégation par max pooling
    for idx in range(batch_size):
        sample_mask = (nd_to_sample == idx)
        if not sample_mask.any():
            continue
            
        sample_masks = room_masks[sample_mask]
        global_images[idx] = torch.max(sample_masks, dim=0)[0]
        
    return global_images

# Ajouter cette fonction en haut du fichier
def calculate_fid(real_imgs, fake_imgs, device):
    inception_dim = 299
    
    # Modèle Inception v3
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    model.to(device)
    model.fc = torch.nn.Identity()  # Supprimer la dernière couche
    
    # Fonction de prétraitement
    preprocess = transforms.Compose([
        transforms.Resize((inception_dim, inception_dim)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def get_features(imgs):
        # Convertir les masques en RGB
        imgs = imgs.expand(-1, 3, -1, -1)
        imgs = preprocess(imgs)
        
        with torch.no_grad():
            features = model(imgs)
        return features.cpu().numpy()
    
    # Calculer les caractéristiques
    real_features = get_features(real_imgs)
    fake_features = get_features(fake_imgs)
    
    # Calculer les statistiques
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculer le FID
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2*covmean)
    return fid

def orthogonal_regularization(model):
    reg = 0
    for param in model.parameters():
        if param.dim() > 1:
            # Reshape les tenseurs 4D en matrices 2D
            if param.dim() == 4:
                w = param.view(param.size(0), -1)  # [out_channels, in_channels * kernel_h * kernel_w]
            else:
                w = param
            
            wwt = torch.mm(w, w.t())
            identity = torch.eye(wwt.size(0)).to(wwt.device)
            reg += torch.norm(wwt - identity, p='fro')  # Norme de Frobenius
    
    return reg


lambda_l1 = 1.0
lambda_fm = 5.0

if __name__ == "__main__":
    # Initialisation des dossiers
    exmps_folder = f"{opt.exmps_folder}_{opt.target_set}"
    exp_dir = f"./exmps/{exmps_folder}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs("./checkpoint", exist_ok=True)
    checkpoint_dir = f"./checkpoint/{exmps_folder}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(f"./exmps/{exmps_folder}/samples", exist_ok=True)
    
    # Fichier CSV pour le suivi
    csv_path = f"./exmps/{exmps_folder}/training_history.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Epoch', 'D_loss', 'G_loss', 'Real_validity', 'Fake_validity', 'GP', 'G_adv', 'G_l1', 'G_fm', 'FID'])

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            gain = nn.init.calculate_gain('leaky_relu', 0.1)
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight, 0, 0.02)
            else:
                nn.init.normal_(m.weight, 0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    # Initialisation des modèles
    generator = Generator()
    discriminator = Discriminator()
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Configuration du device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    generator.to(device)
    discriminator.to(device) 

    # Chargement des données avec la nouvelle taille d'image
    transform = transforms.Normalize(mean=[0.5], std=[0.5])
    fp_dataset_train = FloorplanImageDataset(
        opt.data_path, 
        transform, 
        target_set=opt.target_set,
        split='train'
    )

    fp_loader = DataLoader(
        fp_dataset_train,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=floorplan_collate_fn
    )

    fp_dataset_test = FloorplanImageDataset(
        opt.data_path, 
        transform, 
        target_set=opt.target_set, 
        split='eval'
    )

    fp_loader_test = DataLoader(
        fp_dataset_test,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=floorplan_collate_fn
    )

    # Optimiseurs
    optimizer_G = torch.optim.Adam(
        generator.parameters(), 
        lr=opt.g_lr, 
        betas=(opt.b1, opt.b2)
    )

    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), 
        lr=opt.d_lr, 
        betas=(opt.b1, opt.b2)
    )
    
    start_epoch = 0
    start_batch = 0

    if opt.resume:
        if os.path.isfile(opt.resume):
            print(f"Loading checkpoint '{opt.resume}'")
            checkpoint = torch.load(opt.resume, map_location=device)
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            start_epoch = checkpoint['epoch'] + 1
            start_batch = checkpoint['batch'] + 1
            print(f"Resuming training from epoch {start_epoch}, batch {start_batch}")
        else:
            print(f"No checkpoint found at '{opt.resume}', starting from scratch")

    # Boucle d'entraînement
    batches_done = start_batch

    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='min', factor=0.5, patience=5
    )
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=5
    )

    val_images = []
    for i, batch in enumerate(fp_loader_test):
        if len(val_images) >= 100: 
            break
        
        # Extraction des données du batch
        rooms_mks = batch['masks']
        num_nodes = batch['num_nodes']
        
        # Création de nd_to_sample
        nd_to_sample = torch.repeat_interleave(
            torch.arange(len(num_nodes)), 
            num_nodes
        ).to(device)
        
        # Transfert des masques sur le device
        mks = rooms_mks.unsqueeze(1).to(device)
        batch_size = len(num_nodes)
        
        # Calcul de real_global pour ce batch
        real_global = compose_global_image(mks, nd_to_sample, batch_size, target_size=256)
        real_global = 2 * real_global - 1  # Normalisation [-1,1]
        
        val_images.append(real_global.cpu())

    val_images = torch.cat(val_images)[:100].to(device)
    
    for epoch in range(start_epoch, opt.n_epochs):
        d_loss_epoch = []
        g_loss_epoch = []
        real_val_epoch = []
        fake_val_epoch = []
        gp_epoch = []
        g_adv_epoch = []
        g_l1_epoch = []
        g_fm_epoch = []
        fid_value = 0.0
        
        for i, batch in enumerate(fp_loader):
            rooms_mks = batch['masks']
            graph_nodes = batch['node_features']
            graph_edges = batch['edge_index']
            num_nodes = batch['num_nodes']

            # Vérification de cohérence des données
            total_rooms = sum(num_nodes)
            if total_rooms != rooms_mks.size(0):
                raise RuntimeError(
                    f"Incohérence: Nombre total de pièces ({total_rooms}) "
                    f"ne correspond pas aux masques ({rooms_mks.size(0)})"
                )
            if total_rooms != graph_nodes.size(0):
                raise RuntimeError(
                    f"Incohérence: Nombre total de pièces ({total_rooms}) "
                    f"ne correspond pas aux nœuds ({graph_nodes.size(0)})"
                )

            nd_to_sample = torch.repeat_interleave(
                torch.arange(len(num_nodes)), 
                num_nodes
            ).to(device)

            mks = rooms_mks.unsqueeze(1).to(device)
            nds = graph_nodes.to(device)
            eds = graph_edges.to(device)
            batch_size = len(num_nodes)

            given_y_global_real = add_pool(nds, nd_to_sample, batch_size)
            # ---------------------
            #  Entraînement du discriminateur
            # ---------------------
            optimizer_D.zero_grad()

            # Image globale réelle
            real_global = compose_global_image(mks, nd_to_sample, batch_size)
            real_global = 2 * real_global - 1
            real_global = real_global + 0.05 * torch.randn_like(real_global)

            # Génération image globale fake
            z = torch.randn(nd_to_sample.max().item() + 1, 128, device=device)
            z_rooms = z[nd_to_sample] + 0.1 * torch.randn_like(z[nd_to_sample])
            gen_rooms = generator(z_rooms, mks, nds, eds, nd_to_sample)

            fake_global = compose_global_image(gen_rooms, nd_to_sample, batch_size)
            fake_global = 2 * fake_global - 1
            fake_global = fake_global + 0.05 * torch.randn_like(fake_global)

            # Évaluation par le discriminateur
            real_validity = discriminator(real_global, given_y_global_real, None)
            fake_validity = discriminator(fake_global.detach(), given_y_global_real, None)

            # real_validity = torch.clamp(real_validity, -5, 5)
            # fake_validity = torch.clamp(fake_validity, -5, 5)

            gradient_penalty = compute_gradient_penalty(
                discriminator,
                real_global.data,
                fake_global.data,
                given_y_global_real,
                None
            )

            # Calcul et rétropropagation de la loss
            d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + opt.lambda_gp * gradient_penalty
            d_loss.backward()
            for p in discriminator.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-0.1, 0.1)
            optimizer_D.step()
            # Stockage des métriques
            d_loss_epoch.append(d_loss.item())
            real_val_epoch.append(torch.mean(real_validity).item())
            fake_val_epoch.append(torch.mean(fake_validity).item())
            gp_epoch.append(gradient_penalty.item())
            
            # -----------------
            #  Entraînement du générateur
            # -----------------
            if batches_done % opt.n_critic == 0:
                optimizer_G.zero_grad()
                
                fake_validity = discriminator(fake_global, given_y_global_real, None)
                g_loss_adv = -torch.mean(fake_validity)
                g_loss_l1 = F.l1_loss(gen_rooms, (mks + 1)/2) 
                with torch.no_grad():
                    real_features = discriminator.feature_extractor(real_global, given_y_global_real, None)
                    fake_features = discriminator.feature_extractor(fake_global, given_y_global_real, None)
                g_loss_fm = F.l1_loss(fake_features, real_features)
                g_loss_orth = 0.0
                if batches_done % 10 == 0:
                    g_loss_orth = orthogonal_regularization(generator)
                g_loss = g_loss_adv + lambda_l1 * g_loss_l1 + lambda_fm * g_loss_fm + 0.01 * g_loss_orth
                g_loss.backward()
                for p in generator.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-0.05, 0.05)
                optimizer_G.step()

                g_adv_epoch.append(g_loss_adv.item())
                g_l1_epoch.append(g_loss_l1.item())
                g_fm_epoch.append(g_loss_fm.item())
                g_loss_epoch.append(g_loss.item())
            
            batches_done += 1
            
            # Affichage console
            if i % 10 == 0:
                current_g_loss = g_loss.item() if batches_done % opt.n_critic == 0 else 0
                print(
                    f"[Epoch {epoch}/{opt.n_epochs}] "
                    f"[Batch {i}/{len(fp_loader)}] "
                    f"D: {d_loss.item():.6f} | "
                    f"G: {current_g_loss:.6f} | "
                    f"RealV: {torch.mean(real_validity).item():.4f} | "
                    f"FakeV: {torch.mean(fake_validity).item():.4f} | "
                    f"GP: {gradient_penalty.item():.4f}"
                )

        # Calcul des moyennes pour l'époque
        d_loss_mean = np.mean(d_loss_epoch)
        g_loss_mean = np.mean(g_loss_epoch) if g_loss_epoch else 0
        real_val_mean_epoch = np.mean(real_val_epoch)
        fake_val_mean_epoch = np.mean(fake_val_epoch)
        gp_mean = np.mean(gp_epoch)

        scheduler_D.step(d_loss_mean)
        scheduler_G.step(g_loss_mean)
        
        print(
            f"[Epoch {epoch}/{opt.n_epochs}] "
            f"D_loss: {d_loss_mean:.6f} | "
            f"G_loss: {g_loss_mean:.6f} | "
            f"RealV: {real_val_mean_epoch:.4f} | "
            f"FakeV: {fake_val_mean_epoch:.4f} | "
            f"GP: {gp_mean:.4f}"
        )
        
        # Sauvegarde dans le CSV
        with open(csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                epoch,
                d_loss_mean,
                g_loss_mean,
                real_val_mean_epoch,
                fake_val_mean_epoch,
                gp_mean,
                np.mean(g_adv_epoch) if g_adv_epoch else 0,
                np.mean(g_l1_epoch) if g_l1_epoch else 0,
                np.mean(g_fm_epoch) if g_fm_epoch else 0,
                fid_value if epoch >= 20 else 100
            ])
        
        # Sauvegarde des images globales
        if epoch % opt.sample_interval == 0:
            generator.eval()
            with torch.no_grad():
                try:
                    test_batch = next(iter(fp_loader_test))
                except StopIteration:
                    fp_loader_test = DataLoader(fp_dataset_test, batch_size=8, shuffle=True, collate_fn=floorplan_collate_fn)
                    test_batch = next(iter(fp_loader_test))
                
                # Extraction des données du batch (5 éléments)
                rooms_mks = test_batch['masks']
                graph_nodes = test_batch['node_features']
                graph_edges = test_batch['edge_index']
                num_nodes = test_batch['num_nodes']
                # num_edges = test_batch['num_edges']
                
                # Construire nd_to_sample à partir de num_nodes
                nd_to_sample = torch.repeat_interleave(
                    torch.arange(len(num_nodes)), 
                    num_nodes
                ).to(device)
                
                # Transfert sur le device
                mks_test = rooms_mks.unsqueeze(1).to(device)
                nds_test = graph_nodes.to(device)
                eds_test = graph_edges.to(device)
                # Calcul du batch_size
                batch_size = len(num_nodes)
                
                # Génération du bruit
                z_test = torch.randn(nd_to_sample.max().item() + 1, 128, device=device)
                z_rooms_test = z_test[nd_to_sample]
                
                # Génération des masques de pièces
                gen_rooms_test = generator(z_rooms_test, mks_test, nds_test, eds_test, nd_to_sample)
                
                # Composition des images globales
                real_global_test = compose_global_image(mks_test, nd_to_sample, batch_size, 256)
                fake_global_test = compose_global_image(gen_rooms_test, nd_to_sample, batch_size, 256)

                real_global_test = 2 * real_global_test - 1
                fake_global_test = 2 * fake_global_test - 1
                
                # Normalisation et sauvegarde
                real_global_normalized = (real_global_test + 1) / 2
                fake_global_normalized = (fake_global_test + 1) / 2

                if epoch >= 20 and len(real_global_test) >= 2:

                    try:
                        fid_value = calculate_fid(
                            real_global_test, 
                            fake_global_test, 
                            device
                        )
                        print(f"FID: {fid_value:.2f}")
                    except Exception as e:
                        print(f"Erreur dans le calcul FID: {str(e)}")
                        fid_value = 100
                elif epoch < 20:  # Valeur par défaut haute
                        print("FID non calculé (époques < 20)")

                else:
                    print(f"Pas assez d'échantillons pour FID (disponible: {len(real_global_test)})")


                # with torch.no_grad():
                #     fake_sample = generator.sample(100)  # À implémenter
                #     fid_value = calculate_fid(val_images, fake_sample, device)
                
                save_image(
                    real_global_normalized, 
                    f"./exmps/{exmps_folder}/samples/epoch_{epoch}_real.png", 
                    nrow=4,
                    padding=2,
                    normalize=False
                )
                save_image(
                    fake_global_normalized, 
                    f"./exmps/{exmps_folder}/samples/epoch_{epoch}_fake.png", 
                    nrow=4,
                    padding=2,
                    normalize=False
                )
                
                print(f"Saved samples at epoch {epoch}")
            
            generator.train()
        
        # Sauvegarde des checkpoints
        if epoch % 100 == 0 or epoch == opt.n_epochs - 1:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch,
                'batch': batches_done
            }, f'{checkpoint_dir}/global_epoch_{epoch}.pth')
            
            torch.save(
                generator.state_dict(),
                f'{checkpoint_dir}/generator_epoch_{epoch}.pth'
            )
            
            print(f"Saved checkpoints at epoch {epoch}")

    # scheduler_D.step(d_loss_mean)
    # scheduler_G.step(g_loss_mean)

    # Sauvegarde finale
    torch.save(
        generator.state_dict(), 
        f'{checkpoint_dir}/generator_final.pth'
    )
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
        'epoch': epoch,
        'batch': batches_done
    }, f'{checkpoint_dir}/global_final.pth')
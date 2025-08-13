import torch
import numpy as np
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy import linalg
import torch.nn.functional as F
from model import Generator
from floorplan_maps import FloorplanGraphDataset, floorplan_collate_fn
from utils import _init_input, draw_masks
import matplotlib.pyplot as plt
import argparse

# Inception v3 pour FID
try:
    from torchvision.models import inception_v3
    INCEPTION_AVAILABLE = True
except ImportError:
    print("Inception v3 non disponible, FID ne peut pas être calculé")
    INCEPTION_AVAILABLE = False

class FIDCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        if INCEPTION_AVAILABLE:
            self.inception_model = inception_v3(
                weights='DEFAULT',  
                transform_input=False,
                aux_logits=True   
            )
            # Remplacer la couche finale
            self.inception_model.fc = torch.nn.Identity()
            self.inception_model.eval()
            self.inception_model.to(device)
        else:
            self.inception_model = None
    
    def get_inception_features(self, images):
        if self.inception_model is None:
            raise ValueError("Inception model non disponible")
        
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        features_list = []
        
        for img in images:
            # Convertir en PIL Image si nécessaire
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            # Appliquer les transformations
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.inception_model(img_tensor)
                features_list.append(features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
    
    def calculate_fid(self, real_features, fake_features):
        """Calcule le score FID"""
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
        
        # Calcul de la distance Fréchet
        diff = mu_real - mu_fake
        covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        return fid

def compose_global_image(masks, nd_to_sample, batch_size, target_size=256):
    """Compose une image globale à partir des masques de pièces (version compatible)"""
    # Vérification des dimensions
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    
    # Redimensionnement initial si nécessaire
    if masks.size(2) != 32 or masks.size(3) != 32:
        masks = F.interpolate(masks, size=(32, 32), mode='bilinear', align_corners=False)
    
    device = masks.device
    global_images = torch.zeros(batch_size, 1, 32, 32, device=device)
    
    # Agrégation par max pooling
    for idx in range(batch_size):
        sample_mask = (nd_to_sample == idx)
        if not sample_mask.any():
            continue
            
        sample_masks = masks[sample_mask]
        global_images[idx] = torch.max(sample_masks, dim=0)[0]
    
    # Redimensionnement final
    if target_size != 32:
        global_images = F.interpolate(
            global_images, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        )
    
    return global_images

def generate_plans_for_fid(model, dataset, num_samples=1000):
    """Génère des plans pour l'évaluation FID avec composition globale"""
    model.eval()
    generated_images = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=floorplan_collate_fn)
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # Vérification du format des données (dict)
            if isinstance(sample, dict):
                mks = sample['masks']
                nds = sample['node_features']
                eds = sample['edge_index']
                num_nodes = sample['num_nodes']
            else:
                print(f"Format de sample inattendu: {type(sample)}")
                continue
                
            # Conversion en tenseurs si nécessaire
            if not isinstance(mks, torch.Tensor):
                try:
                    mks = torch.tensor(mks, dtype=torch.float32)
                except:
                    print("Échec de conversion de mks en tenseur")
                    continue
            
            # Création de nd_to_sample
            nd_to_sample = torch.repeat_interleave(
                torch.arange(len(num_nodes)), 
                num_nodes
            ).long()
            
            num_samples_in_batch = 1
            z = torch.randn(num_samples_in_batch, model.latent_dim)
            
            try:
                # Génération des masques de pièces
                masks = model(z, mks, nds, eds, nd_to_sample)
                
                # Composition de l'image globale
                global_img = compose_global_image(
                    masks, 
                    nd_to_sample, 
                    batch_size=len(num_nodes),
                    target_size=256
                )
                
                # CORRECTION: Gestion des dimensions
                # Supprimer toutes les dimensions de taille 1
                img_np = global_img.squeeze().cpu().numpy()
                
                # Vérifier le nombre de dimensions
                if img_np.ndim > 2:
                    # Si on a une dimension supplémentaire, prendre le premier élément
                    img_np = img_np[0] if img_np.shape[0] == 1 else img_np
                
                # Convertir en uint8 et dupliquer en RGB
                img_np = (img_np * 255).astype(np.uint8)
                
                # Si l'image est en niveaux de gris (2D), la convertir en RGB
                if img_np.ndim == 2:
                    img_np = np.stack([img_np]*3, axis=-1)
                
                global_img_pil = Image.fromarray(img_np, 'RGB')
                generated_images.append(global_img_pil)
            except Exception as e:
                print(f"Erreur lors de la génération: {e}")
                continue
                
    return generated_images

def get_real_plans_for_fid(dataset, num_samples=1000):
    """Récupère les plans réels pour l'évaluation FID avec composition globale"""
    real_images = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=floorplan_collate_fn)
    
    for i, sample in enumerate(dataloader):
        if i >= num_samples:
            break
            
        # Vérification du format des données (dict)
        if isinstance(sample, dict):
            mks = sample['masks']
            nds = sample['node_features']
            num_nodes = sample['num_nodes']
        else:
            print(f"Format de sample inattendu: {type(sample)}")
            continue
            
        # Conversion en tenseurs si nécessaire
        if not isinstance(mks, torch.Tensor):
            try:
                mks = torch.tensor(mks, dtype=torch.float32)
            except:
                print("Échec de conversion de mks en tenseur")
                continue
                
        try:
            # Création de nd_to_sample
            nd_to_sample = torch.repeat_interleave(
                torch.arange(len(num_nodes)), 
                num_nodes
            ).long()
            
            # Composition de l'image globale
            global_img = compose_global_image(
                mks.unsqueeze(1), 
                nd_to_sample, 
                batch_size=len(num_nodes),
                target_size=256
            )
            
            # CORRECTION: Gestion des dimensions
            # Supprimer toutes les dimensions de taille 1
            img_np = global_img.squeeze().cpu().numpy()
            
            # Vérifier le nombre de dimensions
            if img_np.ndim > 2:
                # Si on a une dimension supplémentaire, prendre le premier élément
                img_np = img_np[0] if img_np.shape[0] == 1 else img_np
            
            # Convertir en uint8 et dupliquer en RGB
            img_np = (img_np * 255).astype(np.uint8)
            
            # Si l'image est en niveaux de gris (2D), la convertir en RGB
            if img_np.ndim == 2:
                img_np = np.stack([img_np]*3, axis=-1)
            
            global_img_pil = Image.fromarray(img_np, 'RGB')
            real_images.append(global_img_pil)
        except Exception as e:
            print(f"Erreur lors de la composition du plan réel: {e}")
            continue
        
    return real_images

def evaluate_checkpoint_fid(checkpoint_path, dataset, num_samples=1000):
    """Évalue le FID d'un checkpoint spécifique"""
    print(f"Évaluation FID pour: {checkpoint_path}")
    # Charger le modèle
    model = Generator()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Charger avec gestion d'erreur améliorée
    state_dict = state_dict.get('generator', state_dict)
    model_dict = model.state_dict()

    # Filtrer les clés compatibles
    pretrained_dict = {k: v for k, v in state_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Chargé {len(pretrained_dict)}/{len(model_dict)} poids du modèle")
    
    model.eval()
    # Calculer FID
    fid_calculator = FIDCalculator()
    # Générer des plans
    print("Génération des plans...")
    generated_plans = generate_plans_for_fid(model, dataset, num_samples)
    print(f"Nombre d'images générées : {len(generated_plans)}")
    
    # Vérifier si nous avons suffisamment d'images générées
    if len(generated_plans) < 10:  # Seuil plus bas
        print(f"Trop peu d'images générées ({len(generated_plans)}), impossible de calculer le FID")
        return float('nan')
    
    # Récupérer les plans réels
    print("Récupération des plans réels...")
    real_plans = get_real_plans_for_fid(dataset, num_samples)
    print(f"Nombre d'images réelles : {len(real_plans)}")
    
    # Vérification des images
    if len(generated_plans) < 10 or len(real_plans) < 10:
        print(f"Erreur : images insuffisantes (générées: {len(generated_plans)}, réelles: {len(real_plans)})")
        return float('nan')
    
    # Calculer les features Inception
    print("Calcul des features Inception...")
    try:
        real_features = fid_calculator.get_inception_features(real_plans)
        fake_features = fid_calculator.get_inception_features(generated_plans)
        print(f"Features réels: {real_features.shape}, Features générés: {fake_features.shape}")
    except Exception as e:
        print(f"Erreur lors de l'extraction des features Inception : {e}")
        return float('nan')
    
    # Vérification des features
    if real_features.size == 0 or fake_features.size == 0:
        print("Erreur : features Inception vides, impossible de calculer le FID.")
        return float('nan')
    
    # Calculer FID
    try:
        fid_score = fid_calculator.calculate_fid(real_features, fake_features)
        print(f"FID Score: {fid_score:.2f}")
        return fid_score
    except Exception as e:
        print(f"Erreur lors du calcul du FID : {e}")
        return float('nan')

def evaluate_all_checkpoints(checkpoint_dir, dataset, output_file="fid_results.txt"):
    """Évalue le FID pour tous les checkpoints pertinents"""
    # Trouver tous les checkpoints
    all_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    # Filtrer les checkpoints pertinents
    checkpoint_files = []
    for f in all_files:
        filename = os.path.basename(f)
        # Prendre les modèles finaux
        if filename in ["generator_final.pth", "global_final.pth"]:
            checkpoint_files.append(f)
        # Prendre les epochs spécifiques (tous les 100 epochs)
        elif "epoch" in filename:
            try:
                epoch_num = int(filename.split('_')[-1].split('.')[0])
                # Prendre tous les 100 epochs à partir de 100
                if epoch_num % 100 == 0 and epoch_num >= 100:
                    checkpoint_files.append(f)
                # Prendre toujours la dernière epoch (999)
                elif epoch_num == 999:
                    checkpoint_files.append(f)
            except:
                continue
    
    # Trier par ordre d'epoch
    checkpoint_files.sort(key=lambda x: (int(os.path.basename(x).split('_')[-1].split('.')[0]) 
                                        if "epoch" in x else 9999))
    
    print(f"{len(checkpoint_files)} checkpoints sélectionnés : {checkpoint_files}")
    
    results = []
    for checkpoint_file in checkpoint_files:
        # Identifier le type de checkpoint
        filename = os.path.basename(checkpoint_file)
        if "generator" in filename:
            checkpoint_type = "generator"
        elif "global" in filename:
            checkpoint_type = "global"
        else:
            checkpoint_type = "unknown"
        
        # Extraire le numéro d'epoch
        epoch = "final" if "final" in filename else filename.split('_')[-1].split('.')[0]
        
        try:
            fid_score = evaluate_checkpoint_fid(checkpoint_file, dataset, num_samples=500)
            results.append({
                'type': checkpoint_type,
                'fid': fid_score,
                'epoch': epoch,
                'checkpoint': filename
            })
            print(f"{checkpoint_type.capitalize()} epoch {epoch} - FID = {fid_score:.2f}")
        except Exception as e:
            print(f"Erreur pour {checkpoint_file}: {e}")
            results.append({
                'type': checkpoint_type,
                'fid': float('nan'),
                'epoch': epoch,
                'checkpoint': filename
            })
    
    # Sauvegarder les résultats
    with open(output_file, 'w') as f:
        f.write("Type\tEpoch\tFID Score\tCheckpoint\n")
        for result in results:
            f.write(f"{result['type']}\t{result['epoch']}\t{result['fid']:.2f}\t{result['checkpoint']}\n")
    
    return results

def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description='Évaluation FID pour les modèles de plans d\'étage')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoint", help='Répertoire des checkpoints')
    parser.add_argument('--data_path', type=str, default="data/sample_list.txt", help='Chemin vers le dataset')
    parser.add_argument('--target_set', type=int, default=8, help='Set cible pour le dataset')
    parser.add_argument('--output_file', type=str, default="fid_results.txt", help='Fichier de sortie pour les résultats')
    args = parser.parse_args()
    
    if not INCEPTION_AVAILABLE:
        print("Inception v3 non disponible. Installez torchvision avec inception_v3.")
        return
    
    # Charger le dataset avec les mêmes paramètres que l'entraînement
    dataset = FloorplanGraphDataset(
        args.data_path, 
        transforms.Normalize(mean=[0.5], std=[0.5]), 
        target_set=args.target_set,
        split='eval'  
    )
    
    print("Début de l'évaluation FID pour tous les checkpoints pertinents...")
    results = evaluate_all_checkpoints(args.checkpoint_dir, dataset, args.output_file)
    
    # Affichage des résultats
    if results:
        print("\nRésultats complets :")
        for result in results:
            print(f"- {result['type'].capitalize()} epoch {result['epoch']}: FID = {result['fid']:.2f}")
        
        # Trouver le meilleur modèle (ignorer les NaN)
        valid_results = [r for r in results if not np.isnan(r['fid'])]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['fid'])
            print(f"\nMeilleur modèle: {best_result['type']} epoch {best_result['epoch']} avec FID {best_result['fid']:.2f}")
        else:
            print("\nAucun résultat FID valide trouvé.")
    else:
        print("Aucun checkpoint trouvé.")
    
    print("Évaluation FID terminée !")

if __name__ == "__main__":
    main()
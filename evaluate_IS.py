import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Generator
from floorplan_maps import FloorplanGraphDataset, floorplan_collate_fn
import sys  
import argparse  

def compose_global_image(masks, nd_to_sample, batch_size, target_size=256):
    """Compose une image globale à partir des masques de pièces"""
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    
    # Ajout: vérification des dimensions
    print(f"Entrée compose_global_image: masks shape={masks.shape}")
    
    # Interpolation initiale seulement si nécessaire
    if masks.size(2) != 32 or masks.size(3) != 32:
        masks = F.interpolate(masks, size=(32, 32), mode='bilinear', align_corners=False)
    
    device = masks.device
    global_images = torch.zeros(batch_size, 1, 32, 32, device=device)
    
    for idx in range(batch_size):
        sample_mask = (nd_to_sample == idx)
        if not sample_mask.any():
            continue
        sample_masks = masks[sample_mask]
        global_images[idx] = torch.max(sample_masks, dim=0)[0]
    
    # Correction cruciale: interpolation finale
    if target_size != 32:
        global_images = F.interpolate(
            global_images, 
            size=(target_size, target_size),  # Correction: taille carrée
            mode='bilinear', 
            align_corners=False
        )
    
    # Ajout: log de débogage
    print(f"Sortie compose_global_image: shape={global_images.shape}")
    return global_images

def generate_plans_for_is(model, dataset, num_samples=1000):
    """Génère des images pour le calcul de l'Inception Score"""
    model.eval()
    generated_images = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=floorplan_collate_fn)
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if i >= num_samples:
                break
                
            if isinstance(sample, dict):
                mks = sample['masks']
                nds = sample['node_features']
                eds = sample['edge_index']
                num_nodes = sample['num_nodes']
            else:
                continue
                
            if not isinstance(mks, torch.Tensor):
                mks = torch.tensor(mks, dtype=torch.float32)
            
            nd_to_sample = torch.repeat_interleave(
                torch.arange(len(num_nodes)), 
                num_nodes
            ).long()
            
            num_samples_in_batch = 1
            z = torch.randn(num_samples_in_batch, model.latent_dim)
            
            try:
                masks = model(z, mks, nds, eds, nd_to_sample)
                global_img = compose_global_image(
                    masks, 
                    nd_to_sample, 
                    batch_size=len(num_nodes),
                    target_size=256
                )

                # Vérification de la forme
                print(f"Forme globale: {global_img.shape}")
                
                # Extraction CORRECTE: [batch, channel, height, width]
                if global_img.dim() == 4:
                    img_tensor = global_img[0, 0]  # Premier élément du batch, premier canal
                else:
                    img_tensor = global_img[0]
                
                # Conversion en numpy
                img_np = img_tensor.cpu().numpy()
                
                # Normalisation et conversion
                img_np = (img_np * 255).astype(np.uint8)
                
                # Conversion en PIL pour traitement
                img_pil = Image.fromarray(img_np)
                
                # Redimensionnement si nécessaire
                if img_np.shape != (256, 256):
                    img_pil = img_pil.resize((256, 256), Image.BILINEAR)
                
                # Conversion garantie en RGB
                if img_pil.mode != 'RGB':
                    img_pil = img_pil.convert('RGB')
                
                img_np = np.array(img_pil)
                
                # Sauvegarde debug
                debug_path = f"debug_{i}.png"
                img_pil.save(debug_path)
                print(f"Image finale: shape={img_np.shape}, mode={img_pil.mode}")
                
                generated_images.append(img_np)
                
            except Exception as e:
                print(f"Erreur génération: {e}", file=sys.stderr)
                continue
                
    if generated_images:
        print(f"Génération réussie: {len(generated_images)} images")
        # Analyse des shapes
        unique_shapes = set(img.shape for img in generated_images)
        print(f"Formes d'images générées: {unique_shapes}")
    return generated_images

def load_generator(checkpoint_path):
    """Charge le générateur depuis un checkpoint"""
    try:
        model = Generator()
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Gestion des différentes structures de checkpoint
        if 'generator' in state_dict:
            state_dict = state_dict['generator']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        model_dict = model.state_dict()
        
        # Filtrage des clés compatibles
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
            else:
                print(f"Clé ignorée: {k} (forme: {v.shape} vs {model_dict[k].shape if k in model_dict else 'clé absente'})")
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Chargé {len(pretrained_dict)}/{len(model_dict)} poids depuis {checkpoint_path}")
        model.eval()
        return model
    except Exception as e:
        print(f"ERREUR chargement modèle {checkpoint_path}: {e}", file=sys.stderr)
        return None

def inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=10):
    """Calcule l'Inception Score pour des images numpy"""
    for i, img in enumerate(imgs):
        if img.shape != (256, 256, 3):
            print(f"Image {i}: forme incorrecte {img.shape}. Correction...")
            from PIL import Image
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((256, 256), Image.BILINEAR)
            imgs[i] = np.array(img_pil)
    if not imgs:
        print("Liste d'images vide!", file=sys.stderr)
        return float('nan'), float('nan')
    
    N = len(imgs)
    print(f"Début calcul IS sur {N} images")
    
    dtype = torch.cuda.FloatTensor if cuda and torch.cuda.is_available() else torch.FloatTensor
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Création du dataset
    class GeneratedDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, transform):
            self.imgs = imgs
            self.transform = transform
            
        def __getitem__(self, index):
            img = self.imgs[index]
            return self.transform(img)
            
        def __len__(self):
            return len(self.imgs)
    
    dataset = GeneratedDataset(imgs, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Chargement du modèle Inception
    try:
        inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        inception_model.eval()
        up = nn.Upsample(size=(299, 299), mode='bilinear').to(device) if resize else None
        
        def get_pred(x):
            if resize and x.shape[2] != 299:
                x = up(x)
            x = inception_model(x)
            return F.softmax(x, dim=1).data.cpu().numpy()

        preds = np.zeros((N, 1000))

        for i, batch in enumerate(dataloader, 0):
            batch = batch.to(device)
            batch_size_i = batch.size()[0]
            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

        split_scores = []
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        mean_score = np.mean(split_scores)
        std_score = np.std(split_scores)
        print(f"IS calculé: {mean_score:.2f} ± {std_score:.2f}")
        return mean_score, std_score
    except Exception as e:
        print(f"ERREUR calcul IS: {e}", file=sys.stderr)
        return float('nan'), float('nan')

# --- Évaluation des checkpoints avec logs améliorés --- #
def evaluate_checkpoint_is(checkpoint_path, dataset, num_samples=1000):
    """Évalue l'Inception Score pour un checkpoint"""
    print(f"\n{'='*50}")
    print(f"Calcul IS pour: {checkpoint_path}")
    
    model = load_generator(checkpoint_path)
    if model is None:
        return float('nan'), float('nan')
    
    print("Génération des images...")
    generated_imgs = generate_plans_for_is(model, dataset, num_samples)
    
    if len(generated_imgs) < 100:
        print(f"TROP PEU D'IMAGES: {len(generated_imgs)} < 100, IS non calculé")
        return float('nan'), float('nan')
    
    print(f"Calcul de l'Inception Score sur {len(generated_imgs)} images...")
    return inception_score(
        generated_imgs, 
        cuda=torch.cuda.is_available(),
        batch_size=32,
        resize=True,
        splits=10
    )

def evaluate_all_checkpoints_is(checkpoint_dir, dataset, output_file="is_results.txt"):
    """Évalue l'IS pour tous les checkpoints pertinents"""
    print(f"\nRecherche de checkpoints dans: {checkpoint_dir}")
    all_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    print(f"Fichiers trouvés: {len(all_files)}")
    
    valid_files = []
    for f in all_files:
        name = os.path.basename(f)
        if "generator" in name or "global" in name:
            valid_files.append(f)
    
    # Trier par epoch
    valid_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)
    
    print(f"Checkpoints valides ({len(valid_files)}):")
    for f in valid_files:
        print(f"- {os.path.basename(f)}")
    
    results = []
    for checkpoint_file in valid_files:
        name = os.path.basename(checkpoint_file)
        try:
            epoch = name.split("_")[-1].split(".")[0]
        except:
            epoch = "final"
        
        model_type = "generator" if "generator" in name else "global"
        
        try:
            mean, std = evaluate_checkpoint_is(checkpoint_file, dataset, 500)
            results.append({
                'type': model_type,
                'epoch': epoch,
                'is_mean': mean,
                'is_std': std,
                'checkpoint': name
            })
        except Exception as e:
            print(f"ERREUR évaluation {name}: {e}", file=sys.stderr)
            results.append({
                'type': model_type,
                'epoch': epoch,
                'is_mean': float('nan'),
                'is_std': float('nan'),
                'checkpoint': name
            })
    
    # Sauvegarde des résultats
    with open(output_file, 'w') as f:
        f.write("Type\tEpoch\tIS Mean\tIS Std\tCheckpoint\n")
        for res in results:
            f.write(f"{res['type']}\t{res['epoch']}\t{res['is_mean']:.2f}\t{res['is_std']:.2f}\t{res['checkpoint']}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Calculate Inception Score for floorplan models')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='Dossier des checkpoints')
    parser.add_argument('--data_path', type=str, default="data/sample_list.txt", help='Chemin du dataset')
    parser.add_argument('--target_set', type=int, default=8, help='Set cible du dataset')
    parser.add_argument('--output_file', type=str, default="is_results.txt", help='Fichier de sortie')
    args = parser.parse_args()

    print("\nChargement du dataset...")
    dataset = FloorplanGraphDataset(
        args.data_path,
        transforms.Normalize(mean=[0.5], std=[0.5]),
        target_set=args.target_set,
        split='eval'
    )
    print(f"Dataset chargé: {len(dataset)} échantillons")
    
    results = evaluate_all_checkpoints_is(args.checkpoint_dir, dataset, args.output_file)
    
    # Affichage détaillé des résultats
    print("\n\n" + "="*50)
    print("RÉSULTATS FINAUX DE L'ÉVALUATION IS")
    print("="*50)
    
    if not results:
        print("AUCUN RÉSULTAT DISPONIBLE")
        return
    
    print("\nDétails par checkpoint:")
    for res in results:
        print(f"- {res['checkpoint']}: {res['is_mean']:.2f} ± {res['is_std']:.2f}")
    
    valid_results = [r for r in results if not np.isnan(r['is_mean'])]
    
    if valid_results:
        best = max(valid_results, key=lambda x: x['is_mean'])
        worst = min(valid_results, key=lambda x: x['is_mean'])
        
        print("\n" + "="*50)
        print(f"MEILLEUR MODÈLE: {best['checkpoint']}")
        print(f"IS: {best['is_mean']:.2f} ± {best['is_std']:.2f}")
        
        print("\n" + "="*50)
        print(f"PIRE MODÈLE: {worst['checkpoint']}")
        print(f"IS: {worst['is_mean']:.2f} ± {worst['is_std']:.2f}")
        
        print("\nÉvolution par epoch:")
        for res in sorted(valid_results, key=lambda x: int(x['epoch']) if x['epoch'].isdigit() else 0):
            print(f"Epoch {res['epoch']}: {res['is_mean']:.2f}")
    else:
        print("\nAUCUN SCORE IS VALIDE CALCULÉ")

    print("\nÉvaluation terminée!")

if __name__ == "__main__":
    main()
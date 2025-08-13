import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
from model_gan import Generator
from floorplan import FloorplanGraphDataset, floorplan_collate_fn
from torch.utils.data import DataLoader
from utils import _init_input, draw_masks
import torchvision.transforms as transforms

def load_model(checkpoint_path):
    """Charge un modèle depuis un checkpoint"""
    model = Generator()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # Si le checkpoint contient 'generator_state_dict', on l'utilise
    if 'generator_state_dict' in checkpoint:
        state_dict = checkpoint['generator_state_dict']
    else:
        state_dict = checkpoint  # fallback (rare)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate_sample_plans(model, dataset, num_samples=5):
    """Génère des plans d'exemple à partir du modèle"""
    model.eval()
    plans = []
    
    # Charger quelques échantillons du dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=floorplan_collate_fn)
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if i >= num_samples:
                break
                
            mks, nds, eds, _, _ = sample
            real_nodes = np.where(nds.detach().cpu()==1)[-1]
            graph = [nds, eds]
            
            # Générer le plan
            z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state={})
            print("given_masks_in shape:", given_masks_in.shape)
            if given_masks_in.shape[1] == 2:
                given_masks_in = given_masks_in[:, :1, :, :]
            masks = model(z, given_masks_in, given_nds, given_eds)
            masks_np = masks.detach().cpu().numpy()
            print(f"Masque généré (min/max): {masks_np.min()} / {masks_np.max()}")
            # Dessiner le plan
            plan_img = draw_masks(masks_np.copy(), real_nodes)
            plans.append({
                'plan': plan_img,
                'real_nodes': real_nodes,
                'graph': graph
            })
    
    return plans

def visualize_checkpoint(checkpoint_path, dataset, output_dir, epoch_num):
    """Visualise les résultats d'un checkpoint spécifique"""
    print(f"Visualisation du checkpoint: {checkpoint_path}")
    
    # Charger le modèle
    model = load_model(checkpoint_path)
    
    # Générer des plans d'exemple
    plans = generate_sample_plans(model, dataset, num_samples=6)
    
    # Créer la figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Plans générés - Époque {epoch_num}', fontsize=16)
    
    for i, plan_data in enumerate(plans):
        row = i // 3
        col = i % 3
        
        plan_img = plan_data['plan']
        real_nodes = plan_data['real_nodes']
        
        axes[row, col].imshow(plan_img)
        axes[row, col].set_title(f'Exemple {i+1}\nTypes: {real_nodes.tolist()}')
        axes[row, col].axis('off')
    
    # Sauvegarder l'image
    output_path = os.path.join(output_dir, f'epoch_{epoch_num}_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Résultats sauvegardés: {output_path}")
    return output_path

def compare_checkpoints(checkpoint_dir, dataset, output_dir):
    """Compare plusieurs checkpoints"""
    # Trouver tous les checkpoints
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "gan_epoch_*.pth"))
    checkpoint_files.sort()
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Trouvé {len(checkpoint_files)} checkpoints")
    
    # Visualiser chaque checkpoint
    for checkpoint_file in checkpoint_files:
        # Extraire le numéro d'époque
        filename = os.path.basename(checkpoint_file)
        epoch_num = filename.split('_')[-1].split('.')[0]
        
        try:
            visualize_checkpoint(checkpoint_file, dataset, output_dir, epoch_num)
        except Exception as e:
            print(f"Erreur lors du traitement de {checkpoint_file}: {e}")
    
    # Créer une comparaison côte à côte
    create_comparison_grid(checkpoint_files, dataset, output_dir)

def create_comparison_grid(checkpoint_files, dataset, output_dir):
    """Crée une grille de comparaison de tous les checkpoints"""
    # Prendre quelques checkpoints représentatifs
    selected_checkpoints = checkpoint_files[::5]  # Un checkpoint sur 5
    if len(selected_checkpoints) > 6:
        selected_checkpoints = selected_checkpoints[:6]
    
    fig, axes = plt.subplots(len(selected_checkpoints), 3, figsize=(15, 5*len(selected_checkpoints)))
    if len(selected_checkpoints) == 1:
        axes = axes.reshape(1, -1)
    
    for i, checkpoint_file in enumerate(selected_checkpoints):
        epoch_num = os.path.basename(checkpoint_file).split('_')[-1].split('.')[0]
        
        try:
            model = load_model(checkpoint_file)
            plans = generate_sample_plans(model, dataset, num_samples=3)
            
            for j, plan_data in enumerate(plans):
                plan_img = plan_data['plan']
                axes[i, j].imshow(plan_img)
                axes[i, j].set_title(f'Époque {epoch_num} - Ex {j+1}')
                axes[i, j].axis('off')
                
        except Exception as e:
            print(f"Erreur pour {checkpoint_file}: {e}")
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'comparison_all_epochs.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparaison sauvegardée: {comparison_path}")

def main():
    # Configuration
    checkpoint_path = "checkpoints/pretrained.pth"
    output_dir = "visualization_results"
    data_path = "data/sample_list.txt"
    
    # Charger le dataset
    dataset = FloorplanGraphDataset(
        data_path, 
        transforms.Normalize(mean=[0.5], std=[0.5]), 
        split='test'
    )
    
    print(f"Visualisation du checkpoint : {checkpoint_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger le modèle
    model = load_model(checkpoint_path)
    
    # Générer des plans d'exemple
    plans = generate_sample_plans(model, dataset, num_samples=6)
    
    # Afficher et sauvegarder les images
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Plans générés - Checkpoint gan_1', fontsize=16)
    
    for i, plan_data in enumerate(plans):
        row = i // 3
        col = i % 3
        plan_img = plan_data['plan']
        axes[row, col].imshow(plan_img)
        axes[row, col].set_title(f'Exemple {i+1}')
        axes[row, col].axis('off')
    
    output_path = os.path.join(output_dir, 'ganerator_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Résultats sauvegardés: {output_path}")

if __name__ == "__main__":
    main() 
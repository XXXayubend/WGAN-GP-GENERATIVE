import matplotlib.pyplot as plt
import numpy as np
import torch
from floorplan_maps import FloorplanGraphDataset, floorplan_collate_fn, BATCH_SIZE
from torch.utils.data import DataLoader
import matplotlib.patches as patches
from utils import ROOM_CLASS, ID_COLOR
import os

# Créer le répertoire de visualisation s'il n'existe pas
os.makedirs("data_visualization", exist_ok=True)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def visualize_sample(original_image, masks, nodes, edges, sample_id):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Sample Visualization - ID: {sample_id}', fontsize=16)
    
    # 1. Original image with bounding boxes and edges
    if original_image is not None:
        axes[0].imshow(original_image)
        axes[0].set_title("Original Floor Plan")
    else:
        axes[0].set_facecolor('white')
        axes[0].set_xlim(0, 256)
        axes[0].set_ylim(0, 256)
        axes[0].set_title("Simulated Original (BB + Edges)")
        axes[0].invert_yaxis()
    
    # Plot bounding boxes
    if nodes.numel() > 0:
        room_types = torch.argmax(nodes, dim=1) if nodes.dim() > 1 else nodes
        
        for i in range(nodes.shape[0]):
            if nodes.shape[1] >= 5:  # Ensure we have BB info
                x, y, w, h = nodes[i, 1].item(), nodes[i, 2].item(), nodes[i, 3].item(), nodes[i, 4].item()
                color_hex = ID_COLOR.get(room_types[i].item(), '#808080')
                color_rgb = hex_to_rgb(color_hex)
                rect = patches.Rectangle(
                    (x, y), w, h, 
                    linewidth=2, 
                    edgecolor=color_rgb,
                    facecolor='none'
                )
                axes[0].add_patch(rect)
                axes[0].text(x, y, f"{i}: {ROOM_CLASS.get(room_types[i].item(), 'Unknown')}", 
                            color='black', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot edges with index validation
    if edges.numel() > 0:
        for edge in edges:
            if len(edge) >= 3:
                i_idx, rel, j_idx = edge[0], edge[1], edge[2]
                i_idx = i_idx.item() if torch.is_tensor(i_idx) else i_idx
                j_idx = j_idx.item() if torch.is_tensor(j_idx) else j_idx
                
                # Skip invalid indices
                if i_idx < 0 or i_idx >= nodes.shape[0] or j_idx < 0 or j_idx >= nodes.shape[0]:
                    continue
                    
                if nodes.shape[1] >= 5:  # Ensure we have BB info
                    x1 = nodes[i_idx, 1].item() + nodes[i_idx, 3].item()/2
                    y1 = nodes[i_idx, 2].item() + nodes[i_idx, 4].item()/2
                    x2 = nodes[j_idx, 1].item() + nodes[j_idx, 3].item()/2
                    y2 = nodes[j_idx, 2].item() + nodes[j_idx, 4].item()/2
                    
                    color = 'green' if rel == 1 else 'red'
                    axes[0].plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
    
    # 2. Generated 32x32 masks
    if masks.numel() > 0:
        combined_mask = np.zeros((32, 32, 3))
        for i in range(len(masks)):
            mask = masks[i].squeeze().numpy()
            room_type = room_types[i].item() if nodes.numel() > 0 else 0
            color_hex = ID_COLOR.get(room_type, '#808080')
            color_rgb = hex_to_rgb(color_hex)
            
            for c in range(3):
                combined_mask[..., c] = np.where(
                    mask > 0.5, 
                    color_rgb[c] * 0.7 + combined_mask[..., c] * 0.3,
                    combined_mask[..., c]
                )
        
        axes[1].imshow(combined_mask)
        axes[1].set_title("Generated Masks (32x32)")
    else:
        axes[1].text(0.5, 0.5, "No masks", ha='center', va='center')
    
    # 3. Room type representation
    axes[2].set_title("Room Type Representation")
    if nodes.numel() > 0:
        room_types = torch.argmax(nodes, dim=1) if nodes.dim() > 1 else nodes
        type_counts = {}
        for room_type in room_types:
            rtype = room_type.item()
            type_counts[rtype] = type_counts.get(rtype, 0) + 1
        
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [f"{ROOM_CLASS.get(t, f'Unknown ({t})')} ({count})" 
                for t, count in sorted_types]
        sizes = [count for _, count in sorted_types]
        colors = [hex_to_rgb(ID_COLOR.get(t, '#808080')) for t, _ in sorted_types]
        
        if sizes:
            wedges, _ = axes[2].pie(
                sizes, 
                colors=colors,
                startangle=90,
                wedgeprops=dict(width=0.4, edgecolor='w')
            )
            axes[2].legend(
                wedges, 
                labels,
                title="Room Types",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
        else:
            axes[2].text(0.5, 0.5, "No rooms", ha='center', va='center')
    else:
        axes[2].text(0.5, 0.5, "No room data", ha='center', va='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"data_visualization/sample_{sample_id}.png")
    plt.close()

def main():
    DATA_PATH = "data/sample_list.txt"
    TARGET_SET = 8
    
    dataset = FloorplanGraphDataset(
        data_path=DATA_PATH,
        target_set=TARGET_SET,
        split='train'
    )
    if len(dataset) == 0:
        print("ERROR: No samples loaded! Check data path and filters")
        print(f"Data path: {DATA_PATH}")
        print(f"Target set: {TARGET_SET}")
        return 
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=floorplan_collate_fn,
        shuffle=True,
        drop_last=False
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: {BATCH_SIZE}, {len(dataloader)} batches")
    
    samples_visualized = 0
    
    for batch_idx, batch in enumerate(dataloader):
        masks = batch['masks']
        nodes = batch['node_features']
        edges = batch['edge_index']
        num_nodes = batch['num_nodes']
        num_edges = batch['num_edges']
        
        print("\nBatch stats:")
        print(f"  Masks: {masks.shape} ({(masks > 0).float().mean().item():.2%} non-zero)")
        print(f"  Nodes: {nodes.shape}")
        print(f"  Edges: {edges.shape}")
        print(f"  Nodes per sample: {num_nodes.tolist()}")
        print(f"  Edges per sample: {num_edges.tolist()}")
        
        # CORRECTION: Calculer les indices de début/fin correctement
        start_indices = [0]
        for i in range(len(num_nodes) - 1):
            start_indices.append(start_indices[-1] + num_nodes[i].item())
        
        # Visualiser chaque échantillon dans le batch
        for i, n_nodes in enumerate(num_nodes):
            start_idx = start_indices[i]
            end_idx = start_idx + n_nodes.item()
            
            sample_masks = masks[start_idx:end_idx]
            sample_nodes = nodes[start_idx:end_idx]
            
            # Filtrer les arêtes pour cet échantillon
            edge_mask = (edges[:, 0] >= start_idx) & (edges[:, 0] < end_idx) & \
                        (edges[:, 2] >= start_idx) & (edges[:, 2] < end_idx)
            sample_edges = edges[edge_mask].clone()
            
            # Convertir les indices globaux en indices locaux
            sample_edges[:, 0] -= start_idx
            sample_edges[:, 2] -= start_idx
            
            # Obtenir l'image originale
            original_img = None
            try:
                if hasattr(dataset, 'get_original_image'):
                    original_img = dataset.get_original_image(batch_idx * BATCH_SIZE + i)
            except Exception as e:
                print(f"Error getting original image: {e}")
            
            # Visualiser l'échantillon
            sample_id = batch_idx * BATCH_SIZE + i
            visualize_sample(
                original_img,
                sample_masks,
                sample_nodes,
                sample_edges,
                sample_id
            )
            samples_visualized += 1
            print(f"Visualized sample {sample_id}")
            
            if samples_visualized >= 3:
                break
        
        if samples_visualized >= 3:
            break

if __name__ == "__main__":
    main()
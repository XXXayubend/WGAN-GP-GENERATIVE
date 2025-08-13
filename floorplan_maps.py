import json, os, random, math
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from utils import ROOM_CLASS, ID_COLOR

# Constantes globales
BATCH_SIZE = 56
TARGET_SIZE = (32, 32)

def filter_graphs(graphs, min_h=0.03, min_w=0.03):
    """Filtre les graphes invalides avec vérification des dimensions"""
    new_graphs = []
    for g in graphs:
        rooms_type = g[0]
        rooms_bbs = g[1]
        
        # Vérifier les échantillons corrompus
        if len(rooms_type) == 0 or None in rooms_bbs or 0 in rooms_type:
            continue
            
        # Vérifier les dimensions minimales
        valid = True
        for bb in rooms_bbs:
            h, w = bb[3]-bb[1], bb[2]-bb[0]
            if h < min_h or w < min_w:
                valid = False
                break
                
        if valid:
            new_graphs.append(g)
            
    return new_graphs

class FloorplanGraphDataset(Dataset):
    def __init__(self, data_path, transform=None, target_set=8, split='train'):
        super().__init__()
        self.split = split
        self.subgraphs = []
        self.target_set = target_set
        base_dir = os.path.dirname(data_path)
        
        # Vérifier l'existence du fichier de données
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, "r") as f1:
            lines = f1.readlines()
        
        # Compter les échantillons par type de split
        count_by_split = defaultdict(int)
        room_counts = defaultdict(int)
        
        for line in lines:
            json_file = line.strip()
            if not json_file:
                continue
                
            full_path = os.path.join(base_dir, "json", json_file)
            
            if not os.path.exists(full_path):
                print(f"JSON file not found: {full_path}")
                continue

            try:
                rms_type, fp_eds, rms_bbs, eds_to_rms, _, valid_indices = reader(full_path)
                
                # Validation des données
                if len(rms_type) == 0 or len(rms_bbs) == 0 or len(fp_eds) == 0:
                    print(f"Empty data in: {full_path}")
                    continue
                
                # Compter les pièces valides (hors portes)
                fp_size = len([x for x in rms_type if x not in [15, 17]])
                room_counts[fp_size] += 1
                
                # Nouvelle logique de filtrage
                if split == 'train':
                    # Toujours ajouter à l'entraînement
                    self.subgraphs.append([rms_type, rms_bbs, fp_eds, eds_to_rms, valid_indices])
                    count_by_split['train'] += 1
                    
                elif split == 'eval' and fp_size == target_set:
                    # Uniquement les échantillons cibles pour l'évaluation
                    self.subgraphs.append([rms_type, rms_bbs, fp_eds, eds_to_rms, valid_indices])
                    count_by_split['eval'] += 1
                    
                elif split == 'test':
                    # Tout pour le test
                    self.subgraphs.append([rms_type, rms_bbs, fp_eds, eds_to_rms, valid_indices])
                    count_by_split['test'] += 1
                    
            except Exception as e:
                print(f"Error processing {full_path}: {str(e)}")
                continue
                
        self.transform = transform
        print(f"\nLoaded {len(self.subgraphs)} graphs for {split} split")
        print(f"Room count distribution: {dict(room_counts)}")
        print(f"Samples by condition: {dict(count_by_split)}")

    def __len__(self):
        return len(self.subgraphs)
    
    def __getitem__(self, index):
        graph = self.subgraphs[index]
        rms_type, rms_bbs, fp_eds, eds_to_rms, valid_indices = graph
        
        # ==================== FILTRAGE DES PORTES ====================
        door_indices = [i for i, t in enumerate(rms_type) if t in [15, 17]]
        if door_indices:
            # Recréer les données sans les portes
            new_rms_type = [t for i, t in enumerate(rms_type) if i not in door_indices]
            new_rms_bbs = [bb for i, bb in enumerate(rms_bbs) if i not in door_indices]
            
            # Mise à jour des arêtes
            new_fp_eds = []
            new_eds_to_rms = []
            for edge, conn in zip(fp_eds, eds_to_rms):
                new_conn = [i for i in conn if i not in door_indices]
                if len(new_conn) >= 2:
                    new_fp_eds.append(edge)
                    # Réindexation des pièces
                    remapped_conn = []
                    for idx in new_conn:
                        new_idx = idx - sum(1 for d in door_indices if d < idx)
                        remapped_conn.append(new_idx)
                    new_eds_to_rms.append(remapped_conn)
            
            rms_type = new_rms_type
            rms_bbs = np.array(new_rms_bbs, dtype=np.float32)
            fp_eds = np.array(new_fp_eds, dtype=np.float32)
            eds_to_rms = new_eds_to_rms
        else:
            rms_bbs = np.array(rms_bbs, dtype=np.float32)
            fp_eds = np.array(fp_eds, dtype=np.float32)

        # ==================== NORMALISATION ====================
        # CORRECTION : Vérifier et ajuster les dimensions des tableaux
        if rms_bbs.size > 0:
            if len(rms_bbs.shape) == 1:
                rms_bbs = rms_bbs.reshape(-1, 4)
            # S'assurer que rms_bbs a 4 colonnes
            if rms_bbs.shape[1] != 4:
                rms_bbs = rms_bbs[:, :4]  # Prendre seulement les 4 premières colonnes
        
        if fp_eds.size > 0:
            if len(fp_eds.shape) == 1:
                fp_eds = fp_eds.reshape(-1, 4)
            # S'assurer que fp_eds a 4 colonnes
            if fp_eds.shape[1] != 4:
                fp_eds = fp_eds[:, :4]  # Prendre seulement les 4 premières colonnes
        
        # CORRECTION : Créer des tableaux vides si nécessaire
        coords_list = []
        
        if rms_bbs.size > 0:
            # Vérifier que rms_bbs a la forme attendue
            if rms_bbs.shape[1] == 4:
                coords_list.append(rms_bbs[:, :2])
                coords_list.append(rms_bbs[:, 2:])
            else:
                print(f"Unexpected rms_bbs shape: {rms_bbs.shape}")
        
        if fp_eds.size > 0:
            # Vérifier que fp_eds a la forme attendue
            if fp_eds.shape[1] == 4:
                coords_list.append(fp_eds[:, :2])
                coords_list.append(fp_eds[:, 2:])
            else:
                print(f"Unexpected fp_eds shape: {fp_eds.shape}")
        
        # Vérifier que tous les tableaux ont 2 colonnes
        valid_coords = [arr for arr in coords_list if arr.shape[1] == 2]
        
        if valid_coords:
            all_coords = np.vstack(valid_coords)
            min_coord = np.min(all_coords, axis=0)
            max_coord = np.max(all_coords, axis=0)
            size = max(max_coord - min_coord) + 1e-8
        else:
            min_coord = np.array([0.0, 0.0])
            size = 1.0
        
        # Appliquer la normalisation
        if rms_bbs.size > 0 and rms_bbs.shape[1] == 4:
            rms_bbs[:, :2] = (rms_bbs[:, :2] - min_coord) / size
            rms_bbs[:, 2:] = (rms_bbs[:, 2:] - min_coord) / size
        
        if fp_eds.size > 0 and fp_eds.shape[1] == 4:
            fp_eds[:, :2] = (fp_eds[:, :2] - min_coord) / size
            fp_eds[:, 2:] = (fp_eds[:, 2:] - min_coord) / size

        # ================= CONSTRUCTION DU GRAPHE =================
        graph_nodes, graph_edges, rooms_mks = self._build_graph(rms_type, rms_bbs, fp_eds, eds_to_rms)
        
        # Conversion en tenseurs
        graph_nodes = one_hot_embedding(graph_nodes)
        node_features = torch.FloatTensor(graph_nodes)
        edge_index = torch.LongTensor(graph_edges)
        rooms_mks = torch.FloatTensor(rooms_mks)
        
        # Normalisation des caractéristiques des nœuds
        num_nodes = len(graph_nodes)
        if num_nodes > 0:
            mean = node_features.mean(dim=0)
            
            if num_nodes > 1:
                std = node_features.std(dim=0) + 1e-8
                node_features = (node_features - mean) / std
            else:
                node_features = node_features - mean
        else:
            node_features = torch.zeros(0, 18)
        
        return {
            'masks': rooms_mks,
            'node_features': node_features,
            'edge_index': edge_index,
            'num_nodes': len(graph_nodes),
            'num_edges': len(graph_edges)
        }

    def generate_room_mask(self, room_idx, fp_eds, eds_to_rms, rms_bbs, out_size=32):
        """Génère un masque avec fallback sur le bounding box"""
        # Trouver les arêtes associées
        eds = []
        for edge_idx, conn in enumerate(eds_to_rms):
            if room_idx in conn:
                eds.append(edge_idx)
        
        # Fallback si pas assez d'arêtes
        if len(eds) < 2:
            return self.bb_to_mask(rms_bbs[room_idx], out_size)
        
        try:
            # Générer le polygone
            poly_sequence = self.make_sequence(np.array([fp_eds[i][:4] for i in eds]))
            if not poly_sequence:
                return self.bb_to_mask(rms_bbs[room_idx], out_size)
                
            # Dessiner le masque
            im_size = 512
            img = Image.new('L', (im_size, im_size), 0)
            draw = ImageDraw.Draw(img)
            poly = [(x * im_size, y * im_size) for x, y in poly_sequence[0]]
            draw.polygon(poly, fill=255)
            
            # Réduction et seuillage
            img = img.resize((out_size, out_size), Image.LANCZOS)
            return np.array(img) > 127
        except:
            return self.bb_to_mask(rms_bbs[room_idx], out_size)
        
    def _build_graph(self, rms_type, rms_bbs, fp_eds, eds_to_rms):
        """Nouvelle implémentation de build_graph utilisant les méthodes existantes"""
        # Initialisation
        nodes = []
        triples = []
        rms_masks = []
        adjacencies = set()
        
        # Création des relations d'adjacence
        for conn in eds_to_rms:
            if len(conn) < 2:
                continue
                
            for i in range(len(conn)):
                for j in range(i+1, len(conn)):
                    idx1, idx2 = min(conn[i], conn[j]), max(conn[i], conn[j])
                    adjacencies.add((idx1, idx2))
        
        # Création des nœuds et des arêtes
        for i, (t, bb) in enumerate(zip(rms_type, rms_bbs)):
            nodes.append(t)
            
            # Génération du masque en utilisant la méthode existante
            mask = self.generate_room_mask(i, fp_eds, eds_to_rms, rms_bbs)
            rms_masks.append(mask)
            
            # Création des relations
            for j in range(i+1, len(rms_type)):
                is_adjacent = 1 if (i, j) in adjacencies else 0
                triples.append([i, is_adjacent, j])
        
        return np.array(nodes), np.array(triples), np.array(rms_masks)

    def bb_to_mask(self, bb, out_size):
        """Convertit un bounding box en masque binaire"""
        mask = np.zeros((out_size, out_size))
        x0, y0, x1, y1 = bb
        
        # Conversion en pixels
        x0 = int(x0 * out_size)
        y0 = int(y0 * out_size)
        x1 = min(out_size, int(x1 * out_size) + 1)
        y1 = min(out_size, int(y1 * out_size) + 1)
        
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
        return mask

    def make_sequence(self, edges, tol=1e-3):
        """Crée une séquence de points fermée avec tolérance numérique"""
        if len(edges) < 2:
            return []
            
        points = []
        for edge in edges:
            points.append(tuple(edge[:2]))
            points.append(tuple(edge[2:]))
            
        # Trouver le point de départ (le plus à gauche)
        start = min(points, key=lambda p: (p[0], p[1]))
        polygon = [start]
        current = start
        
        # Construction du polygone
        while points:
            found = False
            for i, p in enumerate(points):
                if math.isclose(p[0], current[0], abs_tol=tol) and math.isclose(p[1], current[1], abs_tol=tol):
                    # Trouver le point connecté
                    next_point = None
                    if i % 2 == 0:
                        next_point = points[i+1]
                    else:
                        next_point = points[i-1]
                    
                    # Mettre à jour la liste des points
                    if i % 2 == 0:
                        del points[i:i+2]
                    else:
                        del points[i-1:i+1]
                    
                    polygon.append(next_point)
                    current = next_point
                    found = True
                    break
                    
            if not found and points:
                # En cas d'échec, fermer le polygone
                polygon.append(start)
                break
                
        # Fermer le polygone
        if not math.isclose(polygon[0][0], polygon[-1][0], abs_tol=tol) or \
           not math.isclose(polygon[0][1], polygon[-1][1], abs_tol=tol):
            polygon.append(polygon[0])
            
        return [polygon]

def one_hot_embedding(labels, num_classes=18):
    """Encodage one-hot avec exclusion du background (classe 0)"""
    if len(labels) == 0:
        return np.zeros((0, num_classes))
    return np.eye(num_classes + 1)[labels][:, 1:]

def floorplan_collate_fn(batch):
    """Fonction de collage corrigée avec gestion des tenseurs scalaires"""
    # Filtrer les échantillons vides
    valid_batch = [b for b in batch if b['num_nodes'] > 0]
    
    if not valid_batch:
        return {
            'masks': torch.zeros(BATCH_SIZE, *TARGET_SIZE),
            'node_features': torch.zeros(BATCH_SIZE, 18),
            'edge_index': torch.zeros(0, 3, dtype=torch.long),
            'num_nodes': torch.zeros(BATCH_SIZE, dtype=torch.long),
            'num_edges': torch.zeros(BATCH_SIZE, dtype=torch.long)
        }
    
    # Initialisation des listes pour chaque champ
    all_masks = []
    all_node_features = []
    all_edge_index = []
    all_num_nodes = []
    all_num_edges = []
    
    node_offset = 0
    
    for sample in valid_batch:
        # Masks
        all_masks.append(sample['masks'])
        
        # Node features
        all_node_features.append(sample['node_features'])
        
        # Edge index (avec ajustement des indices)
        edges = sample['edge_index'].clone()
        if edges.numel() > 0:
            edges[:, 0] += node_offset
            edges[:, 2] += node_offset
        all_edge_index.append(edges)
        
        # Metadata - conversion en tenseurs scalaires
        all_num_nodes.append(torch.tensor(sample['num_nodes'], dtype=torch.long))
        all_num_edges.append(torch.tensor(sample['num_edges'], dtype=torch.long))
        
        node_offset += sample['num_nodes']
    
    # Concaténation CORRIGÉE
    collated = {
        'masks': torch.cat(all_masks, dim=0),
        'node_features': torch.cat(all_node_features, dim=0),
        'edge_index': torch.cat(all_edge_index, dim=0) if all_edge_index else torch.zeros(0, 3, dtype=torch.long),
        'num_nodes': torch.stack(all_num_nodes),  # Utilisation de stack pour les scalaires
        'num_edges': torch.stack(all_num_edges)   # Utilisation de stack pour les scalaires
    }
    
    # Padding pour atteindre la taille de batch
    current_batch_size = len(valid_batch)
    if current_batch_size < BATCH_SIZE:
        padding_size = BATCH_SIZE - current_batch_size
        
        # Padding pour masks
        collated['masks'] = torch.cat([
            collated['masks'],
            torch.zeros(padding_size, *TARGET_SIZE)
        ], dim=0)
        
        # Padding pour node features
        collated['node_features'] = torch.cat([
            collated['node_features'],
            torch.zeros(padding_size, 18)
        ], dim=0)
        
        # Padding pour edge index
        collated['edge_index'] = torch.cat([
            collated['edge_index'],
            torch.zeros(0, 3, dtype=torch.long)
        ], dim=0)
        
        # Padding pour metadata
        collated['num_nodes'] = torch.cat([
            collated['num_nodes'],
            torch.zeros(padding_size, dtype=torch.long)
        ])
        
        collated['num_edges'] = torch.cat([
            collated['num_edges'],
            torch.zeros(padding_size, dtype=torch.long)
        ])

        total_rooms = collated['masks'].size(0)
        expected_rooms = collated['num_nodes'].sum().item()
        
        if total_rooms != expected_rooms:
            # print(f"Data inconsistency! Rooms: {total_rooms}, Expected: {expected_rooms}")
            # Ajustement automatique
            min_size = min(total_rooms, expected_rooms)
            collated['masks'] = collated['masks'][:min_size]
            collated['node_features'] = collated['node_features'][:min_size]

    return collated


def reader(filename):
    """Lecture robuste des fichiers JSON avec gestion des erreurs améliorée"""
    try:
        with open(filename) as f:
            info = json.load(f)
            
            # Validation des champs
            required = ['boxes', 'edges', 'room_type', 'ed_rm']
            if not all(field in info for field in required):
                print(f"Missing fields in {filename}")
                return [], [], [], [], [], []
            
            # Gestion des dimensions différentes
            rms_type = info['room_type']
            
            # CORRECTION: S'assurer que les boxes ont 4 éléments
            boxes = []
            for box in info['boxes']:
                if len(box) >= 4:
                    boxes.append(box[:4])  # Prendre seulement les 4 premiers éléments
                else:
                    print(f"Invalid box in {filename}: {box}")
                    boxes.append([0,0,1,1])  # Valeur par défaut
            
            rms_bbs = np.array(boxes, dtype=np.float32)
            
            # CORRECTION: S'assurer que les edges ont 4 éléments
            edges = []
            for edge in info['edges']:
                if len(edge) >= 4:
                    edges.append(edge[:4])  # Prendre seulement les 4 premiers éléments
                else:
                    print(f"Invalid edge in {filename}: {edge}")
                    edges.append([0,0,1,1])  # Valeur par défaut
            
            fp_eds = np.array(edges, dtype=np.float32)
            
            eds_to_rms = info['ed_rm']

            
            # Ajuster les dimensions si nécessaire
            min_length = min(len(rms_type), len(rms_bbs))
            if len(rms_type) != min_length or len(rms_bbs) != min_length:
                # print(f"Adjusting dimensions in {filename}: "
                #       f"room_type({len(rms_type)}), boxes({len(rms_bbs)})")
                rms_type = rms_type[:min_length]
                rms_bbs = rms_bbs[:min_length]
            
            # Validation des arêtes
            if len(fp_eds) != len(eds_to_rms):
                print(f"Edge mismatch in {filename}: "
                      f"edges({len(fp_eds)}), ed_rm({len(eds_to_rms)})")
                # Prendre le minimum des deux
                min_edge_length = min(len(fp_eds), len(eds_to_rms))
                fp_eds = fp_eds[:min_edge_length]
                eds_to_rms = eds_to_rms[:min_edge_length]
            
            # Indices valides (hors portes)
            valid_indices = [i for i, t in enumerate(rms_type) if t not in [15, 17]]
            
            return rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms, valid_indices
            
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return [], [], [], [], [], []
import json, os, random, math
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
import matplotlib as plt
import numpy as np
import PIL
import glob
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import random
from utils import ROOM_CLASS, ID_COLOR

def filter_graphs(graphs, min_h=0.03, min_w=0.03):
    new_graphs = []
    for g in graphs:
        
        # retrieve data
        rooms_type = g[0]
        rooms_bbs = g[1]
        
        # discard broken samples
        check_none = np.sum([bb is None for bb in rooms_bbs])
        check_node = np.sum([nd == 0 for nd in rooms_type])
        if (len(rooms_type) == 0) or (check_none > 0) or (check_node > 0):
            continue
		
        # update graph
        new_graphs.append(g)
    return new_graphs

class FloorplanGraphDataset(Dataset):
    def __init__(self, data_path, transform=None, target_set=8, split='train'):
        super(Dataset, self).__init__()
        self.split = split
        self.subgraphs=[]
        self.target_set = target_set
        with open(data_path, "r") as f1:
            lines = f1.readlines()
        h = 0
        for line in lines:
            a = []
            h += 1
            json_path = os.path.join("data", "json", line.strip())
            if split == 'train':
                with open(json_path) as f2:
                    rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp = reader(json_path)
                    fp_size = len([x for x in rms_type if x != 15 and x != 17])
                    if fp_size != target_set:
                        a.append(rms_type)
                        a.append(rms_bbs)
                        a.append(fp_eds)
                        a.append(eds_to_rms)
                        a.append(eds_to_rms_tmp)
                        self.subgraphs.append(a)
                self.augment = True
            elif split == 'eval':
                with open(json_path) as f2:
                    rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp = reader(json_path)
                    fp_size = len([x for x in rms_type if x != 15 and x != 17])
                    if fp_size == target_set:
                        a.append(rms_type)
                        a.append(rms_bbs)
                        a.append(fp_eds)
                        a.append(eds_to_rms)
                        a.append(eds_to_rms_tmp)
                        self.subgraphs.append(a)
                self.augment = False
            elif split == 'test':
                with open(json_path) as f2:
                    rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp = reader(json_path)
                    a.append(rms_type)
                    a.append(rms_bbs)
                    a.append(fp_eds)
                    a.append(eds_to_rms)
                    a.append(eds_to_rms_tmp)
                    self.subgraphs.append(a)
            else:
                print('ERR')
                exit(0)
        self.transform = transform
        print(len(self.subgraphs))   
        
    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, index):
        graph = self.subgraphs[index]
        rms_type = graph[0]
        rms_bbs = graph[1]
        fp_eds = graph[2]
        eds_to_rms = graph[3]
        eds_to_rms_tmp = graph[4]
        
        # Convertir en arrays numpy
        rms_bbs = np.array(rms_bbs)
        fp_eds = np.array(fp_eds)

        # Filtrer les portes (types 15 et 17) AVANT toute transformation
        valid_indices = [i for i, t in enumerate(rms_type) if t not in [15, 17]]
        if not valid_indices:
            # Gérer le cas où il n'y a que des portes (cas rare mais possible)
            return self.__getitem__((index + 1) % len(self))  # Récupérer le prochain élément
        
        rms_type = [rms_type[i] for i in valid_indices]
        rms_bbs = [rms_bbs[i] for i in valid_indices]
        rms_bbs = np.array(rms_bbs)

        # extract boundary box and centralize
        tl = np.min(rms_bbs[:, :2], 0)
        br = np.max(rms_bbs[:, 2:], 0)
        shift = (tl + br) / 2.0 - 0.5
        rms_bbs[:, :2] -= shift
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift
        tl -= shift
        br -= shift
        
        # Mettre à jour eds_to_rms pour ne référencer que les indices valides
        new_eds_to_rms = []
        for edge in eds_to_rms:
            new_edge = [i for i in edge if i in valid_indices]
            if new_edge:  # Ne garder que les edges qui relient des pièces valides
                new_eds_to_rms.append(new_edge)
        
        # build input graph avec les données filtrées
        graph_nodes, graph_edges, rooms_mks = self.build_graph(rms_type, fp_eds, new_eds_to_rms)

        rooms_mks = torch.FloatTensor(rooms_mks)
        graph_nodes = one_hot_embedding(graph_nodes)[:, 1:]
        graph_nodes = torch.FloatTensor(graph_nodes)
        graph_edges = torch.LongTensor(graph_edges)

        # Duplication des canaux pour le générateur
        rooms_mks = rooms_mks.unsqueeze(1)  # [n_rooms, 1, H, W]
        # Redimensionnement des masques pour le générateur
        transform_to_32 = T.Resize((32, 32))
        rooms_mks = transform_to_32(rooms_mks)  # [n_rooms, 1, 32, 32]

        # Génération de l'image complète pour le discriminateur
        full_floorplan = torch.sum(rooms_mks, dim=0, keepdim=True)  # [1, 1, 32, 32]
        full_floorplan = torch.clamp(full_floorplan, 0, 1)

        # Redimensionnement à 64x64
        transform_to_64 = T.Resize((64, 64))
        full_floorplan = transform_to_64(full_floorplan)  # [1, 1, 64, 64]
        full_floorplan = full_floorplan.squeeze(0)  # [1, 64, 64]

        # Normalisation [-1, 1] pour le discriminateur
        full_floorplan = (full_floorplan * 2) - 1

        # Ajout de nd_to_sample pour le pooling par batch
        n_rooms = len(graph_nodes)
        nd_to_sample = torch.LongTensor(n_rooms).fill_(index)

        if self.transform is not None:
            rooms_mks = self.transform(rooms_mks)

        return rooms_mks, graph_nodes, graph_edges, full_floorplan, nd_to_sample


    def draw_masks(self, rms_type, fp_eds, eds_to_rms, im_size=256):
        # import webcolors
        # full_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))  # Semitransparent background.
        rms_masks = []
        fp_mk = np.zeros((32, 32))
        for k in range(len(rms_type)):
            eds = []
            for l, e_map in enumerate(eds_to_rms):
                if k in e_map:
                    eds.append(l)
            rm_im = Image.new('L', (im_size, im_size))
            rm_im = rm_im.filter(ImageFilter.MaxFilter(7))
            dr = ImageDraw.Draw(rm_im)
            poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds]))[0]
            poly = [(im_size*x, im_size*y) for x, y in poly]
            if len(poly) >= 2:
                dr.polygon(poly, fill='white')
            rm_im = rm_im.resize((32, 32)).filter(ImageFilter.MaxFilter(3))
            rm_arr = np.array(rm_im)
            inds = np.where(rm_arr>0)
            fp_mk[inds] = k+1

        # trick to remove overlap
        for k in range(len(rms_type)):
            rm_arr=np.ones((32,32))
            rm_arr = np.zeros((32, 32))
            inds = np.where(fp_mk==k+1)
            rm_arr[inds] = 1.0
            rms_masks.append(rm_arr)

        plt.figure()
        debug_arr = np.sum(np.array(rms_masks), 0)
        debug_arr[debug_arr>0] = 255
        im = Image.fromarray(debug_arr)
        plt.imshow(im)
        plt.show()

        return rms_masks

    def make_sequence(self, edges):
        polys = []
        #print(edges)
        v_curr = tuple(edges[0][:2])
        e_ind_curr = 0
        e_visited = [0]
        seq_tracker = [v_curr]
        find_next = False
        while len(e_visited) < len(edges):
            if find_next == False:
                if v_curr == tuple(edges[e_ind_curr][2:]):
                    v_curr = tuple(edges[e_ind_curr][:2])
                else:
                    v_curr = tuple(edges[e_ind_curr][2:])
                find_next = not find_next 
            else:
                # look for next edge
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        if (v_curr == tuple(e[:2])):
                            v_curr = tuple(e[2:])
                            e_ind_curr = k
                            e_visited.append(k)
                            break
                        elif (v_curr == tuple(e[2:])):
                            v_curr = tuple(e[:2])
                            e_ind_curr = k
                            e_visited.append(k)
                            break

            # extract next sequence
            if v_curr == seq_tracker[-1]:
                polys.append(seq_tracker)
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        v_curr = tuple(edges[0][:2])
                        seq_tracker = [v_curr]
                        find_next = False
                        e_ind_curr = k
                        e_visited.append(k)
                        break
            else:
                seq_tracker.append(v_curr)
        polys.append(seq_tracker)

        return polys

    def flip_and_rotate(self, v, flip, rot, shape=256.):
        v = self.rotate(np.array((shape, shape)), v, rot)
        if flip:
            x, y = v
            v = (shape/2-abs(shape/2-x), y) if x > shape/2 else (shape/2+abs(shape/2-x), y)
        return v
	
    # rotate coords
    def rotate(self, image_shape, xy, angle):
        org_center = (image_shape-1)/2.
        rot_center = (image_shape-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
                -org[0]*np.sin(a) + org[1]*np.cos(a) ])
        new = new+rot_center
        return new

    def build_graph(self, rms_type, fp_eds, eds_to_rms, out_size=64):
        # Filtrer les portes (types 15 et 17)
        valid_indices = [i for i, t in enumerate(rms_type) if t not in [15, 17]]
        nodes = [rms_type[i] for i in valid_indices]
        
        # Créer les arêtes uniquement entre les pièces valides
        triples = []
        for k_idx, k in enumerate(valid_indices):
            for l_idx, l in enumerate(valid_indices):
                if l_idx > k_idx:
                    is_adjacent = any(True for e_map in eds_to_rms 
                                if (k in e_map) and (l in e_map))
                    rel_type = 1 if is_adjacent else -1
                    triples.append([k_idx, rel_type, l_idx])

        # Préparer les données pour les masques
        eds_to_rms_tmp = []
        for l in range(len(eds_to_rms)):                  
            eds_to_rms_tmp.append([eds_to_rms[l][0]])

        rms_masks = []
        im_size = 256
        fp_mk = np.zeros((out_size, out_size))

        for idx, orig_idx in enumerate(valid_indices):  # Correction ici: idx est le nouvel index
            # Récupérer les edges associées
            eds = []
            for l, e_map in enumerate(eds_to_rms_tmp):
                if (orig_idx in e_map):
                    eds.append(l)

            if not eds:
                # Créer un masque par défaut si aucune edge
                rm_im = Image.new('L', (im_size, im_size))
                dr = ImageDraw.Draw(rm_im)
                center = im_size // 2
                size = im_size // 10
                dr.rectangle([center-size, center-size, center+size, center+size], fill='white')
                rm_im = rm_im.resize((out_size, out_size))
                rm_arr = np.array(rm_im)
                rm_arr[rm_arr > 0] = 1.0
                rms_masks.append(rm_arr)
                continue

            # Dessiner la pièce
            rm_im = Image.new('L', (im_size, im_size))
            dr = ImageDraw.Draw(rm_im)
            edge_points = np.array([fp_eds[l][:4] for l in eds])
            
            if len(edge_points) > 0:
                poly = self.make_sequence(edge_points)[0]
                poly = [(im_size*x, im_size*y) for x, y in poly]
                if len(poly) >= 2:
                    dr.polygon(poly, fill='white')
                else:
                    print("Empty room polygon")
            else:
                print("No edge points for room")

            rm_im = rm_im.resize((out_size, out_size))
            rm_arr = np.array(rm_im)
            inds = np.where(rm_arr > 0)
            rm_arr[inds] = 1.0
            rms_masks.append(rm_arr)
            fp_mk[inds] = idx + 1  # Utiliser idx au lieu de k

        # Supprimer les chevauchements
        for idx in range(len(nodes)):
            rm_arr = np.zeros((out_size, out_size))
            inds = np.where(fp_mk == idx+1)
            rm_arr[inds] = 1.0
            rms_masks[idx] = rm_arr

        # Convertir en arrays numpy
        nodes = np.array(nodes)
        triples = np.array(triples)
        rms_masks = np.array(rms_masks)

        return nodes, triples, rms_masks
		
    def build_graph_door_as_dents(self, rms_type, fp_eds, eds_to_rms, out_size=128):

            # create edges
            triples = []
            nodes = [x for x in rms_type if x != 15 and x != 17]

            # doors to rooms
            doors_inds = []
            for k, r in enumerate(rms_type):
                if r in [15, 17]:
                    doors_inds.append(k)

            # for each door compare against all rooms
            door_to_rooms = defaultdict(list)
            for d in doors_inds:
                door_edges = eds_to_rms[d]
                for r in range(len(nodes)):
                    if r not in doors_inds:
                        is_adjacent = any([True for e_map in eds_to_rms if (r in e_map) and (d in e_map)])
                        if is_adjacent:
                            door_to_rooms[d].append(r)


            # encode connections
            for k in range(len(nodes)):
                for l in range(len(nodes)):
                    if l > k:
                        is_adjacent = any([True for d_key in door_to_rooms if (k in door_to_rooms[d_key]) and (l in door_to_rooms[d_key])])
                        if is_adjacent:
                            if 'train' in self.split:
                                triples.append([k, 1, l])
                            else:
                                triples.append([k, 1, l])
                        else:
                            if 'train' in self.split:
                                triples.append([k, -1, l])
                            else:
                                triples.append([k, -1, l])

            # get rooms masks
            eds_to_rms_tmp = []
            for l in range(len(eds_to_rms)):                  
                eds_to_rms_tmp.append([eds_to_rms[l][0]])

            rms_masks = []
            im_size = 256
            # fp_mk = np.zeros((out_size, out_size))

            for k in range(len(nodes)):
                
                # add rooms
                eds = []
                for l, e_map in enumerate(eds_to_rms_tmp):
                    if (k in e_map):
                        eds.append(l)

                # add doors
                eds_door = []
                for d in door_to_rooms:
                    if k in door_to_rooms[d]:
                        door = []
                        for l, e_map in enumerate(eds_to_rms_tmp):
                            if (d in e_map):
                                door.append(l)
                        eds_door.append(door)

                # draw rooms
                rm_im = Image.new('L', (im_size, im_size))
                dr = ImageDraw.Draw(rm_im)
                for eds_poly in [eds]:
                    poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[0]
                    poly = [(im_size*x, im_size*y) for x, y in poly]
                    if len(poly) >= 2:
                        dr.polygon(poly, fill='white')
                    else:
                        print("Empty room")
                        exit(0)

                # draw doors
                doors_im = Image.new('L', (im_size, im_size))
                dr_door = ImageDraw.Draw(doors_im)
                for eds_poly in eds_door:
                    poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[0]
                    poly = [(im_size*x, im_size*y) for x, y in poly]
                    if len(poly) >= 2:
                        dr_door.polygon(poly, fill='white')
                    else:
                        print("Empty room")
                        exit(0)

                doors_im = doors_im.filter(ImageFilter.MinFilter(3)).resize((out_size, out_size))
                doors_arr= np.array(doors_im)
                rm_im = rm_im.filter(ImageFilter.MinFilter(3)).resize((out_size, out_size))
                rm_arr = np.array(rm_im)
                inds = np.where(rm_arr+doors_arr>0)
                rm_arr[inds] = 1.0
                rms_masks.append(rm_arr)

            # convert to array
            nodes = np.array(nodes)
            triples = np.array(triples)
            rms_masks = np.array(rms_masks)

            return nodes, triples, rms_masks

def is_adjacent(box_a, box_b, threshold=0.03):
	
	x0, y0, x1, y1 = box_a
	x2, y2, x3, y3 = box_b

	h1, h2 = x1-x0, x3-x2
	w1, w2 = y1-y0, y3-y2

	xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
	yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0

	delta_x = np.abs(xc2-xc1) - (h1 + h2)/2.0
	delta_y = np.abs(yc2-yc1) - (w1 + w2)/2.0

	delta = max(delta_x, delta_y)

	return delta < threshold

# Ajouter après la classe FloorplanGraphDataset
class FloorplanImageDataset(Dataset):
    def __init__(self, data_path, img_size=128, transform=None, split='train'):
        """
        Dataset pour images de plans d'étage compatibles WGAN-GP
        
        Args:
            data_path: Chemin vers les données
            img_size: Taille des images de sortie
            transform: Transformations torchvision
            split: 'train' ou 'eval'
        """
        self.graph_dataset = FloorplanGraphDataset(
            data_path, 
            transform=None, 
            split=split,
            target_set=8  # Valeur par défaut
        )
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.graph_dataset)
    
    def __getitem__(self, index):
        # Récupérer les données du graphe
        data = self.graph_dataset[index]
        
        # Créer une image RGB vide (fond blanc)
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255
        draw = ImageDraw.Draw(Image.fromarray(img))
        
        # Extraire les données
        rms_type = self.graph_dataset.subgraphs[index][0]
        rms_bbs = self.graph_dataset.subgraphs[index][1]
        fp_eds = self.graph_dataset.subgraphs[index][2]
        
        # Dessiner les pièces
        for i, (t, bb) in enumerate(zip(rms_type, rms_bbs)):
            if t in [15, 17]:  # Ignorer les portes
                continue
                
            # Coordonnées normalisées -> pixels
            x0, y0, x1, y1 = (bb * (self.img_size - 1)).astype(int)
            
            # Obtenir la couleur
            color = ID_COLOR.get(t, (200, 200, 200))
            
            # Dessiner le rectangle
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
        
        # Dessiner les murs
        for edge in fp_eds:
            x1, y1, x2, y2 = (edge * (self.img_size - 1)).astype(int)
            draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=2)
        
        # Convertir en PIL et appliquer les transformations
        img_pil = Image.fromarray(img).convert('L')  # Convertir en niveaux de gris
        return self.transform(img_pil)

def one_hot_embedding(labels, num_classes=19):
	"""Embedding labels to one-hot form.

	Args:
	  labels: (LongTensor) class labels, sized [N,].
	  num_classes: (int) number of classes.

	Returns:
	  (tensor) encoded labels, sized [N, #classes].
	"""
	y = torch.eye(num_classes)
	#print(" label is",labels) 
	return y[labels] 

def floorplan_collate_fn(batch):
    all_rooms_mks = []
    all_nodes = []
    all_edges = []
    all_full_floorplan = []
    all_nd_to_sample = []
    node_offset = 0
    
    for batch_idx, (rooms_mks, nodes, edges, full_floorplan, nd_to_sample) in enumerate(batch):
        # Créer un masque pour les pièces valides
        room_types = torch.argmax(nodes, dim=1)
        valid_mask = (room_types != 15) & (room_types != 17)
        
        # Appliquer le masque
        rooms_mks = rooms_mks[valid_mask]
        nodes = nodes[valid_mask]
        nd_to_sample = nd_to_sample[valid_mask]
        
        # Réindexer les arêtes
        if edges.numel() > 0:
            # Créer un mapping des anciens indices vers les nouveaux
            idx_mapping = {}
            new_idx = 0
            for old_idx in range(valid_mask.size(0)):
                if valid_mask[old_idx]:
                    idx_mapping[old_idx] = new_idx
                    new_idx += 1
            
            # Filtrer et réindexer les arêtes
            valid_edges = []
            for edge in edges:
                old_src = edge[0].item()
                old_dst = edge[2].item()
                if old_src in idx_mapping and old_dst in idx_mapping:
                    new_edge = torch.tensor([
                        idx_mapping[old_src],
                        edge[1].item(),
                        idx_mapping[old_dst]
                    ], dtype=torch.long)
                    valid_edges.append(new_edge)
            
            edges = torch.stack(valid_edges) if valid_edges else torch.empty((0, 3), dtype=torch.long)
        
        # Gestion des échantillons vides
        if rooms_mks.size(0) == 0:
            rooms_mks = torch.zeros(1, 1, 32, 32)
            nodes = torch.zeros(1, 18)
            nd_to_sample = torch.tensor([batch_idx], dtype=torch.long)
            edges = torch.empty((0, 3), dtype=torch.long)
        
        # Accumuler les données
        O = rooms_mks.size(0)
        all_rooms_mks.append(rooms_mks)
        all_nodes.append(nodes)
        all_full_floorplan.append(full_floorplan)
        
        # Création de nd_to_sample
        new_nd_to_sample = torch.full((O,), batch_idx, dtype=torch.long)
        all_nd_to_sample.append(new_nd_to_sample)
        
        # Traitement des arêtes
        if edges.numel() > 0:
            edges = edges.clone()
            edges[:, 0] += node_offset
            edges[:, 2] += node_offset
            all_edges.append(edges)
        else:
            all_edges.append(torch.empty((0, 3), dtype=torch.long))
        
        node_offset += O

    return (
        torch.cat(all_rooms_mks, dim=0),
        torch.cat(all_nodes, dim=0),
        torch.cat(all_edges, dim=0) if all_edges else torch.empty((0, 3), dtype=torch.long),
        torch.stack(all_full_floorplan),
        torch.cat(all_nd_to_sample, dim=0)
    )

def reader(filename):
	with open(filename) as f:
		info =json.load(f)
		rms_bbs=np.asarray(info['boxes'])
		fp_eds=info['edges']
		rms_type=info['room_type']
		eds_to_rms=info['ed_rm']
		s_r=0
		for rmk in range(len(rms_type)):
			if(rms_type[rmk]!=17):
				s_r=s_r+1	
		#print("eds_ro",eds_to_rms)
		rms_bbs = np.array(rms_bbs)/256.0
		fp_eds = np.array(fp_eds)/256.0 
		fp_eds = fp_eds[:, :4]
		tl = np.min(rms_bbs[:, :2], 0)
		br = np.max(rms_bbs[:, 2:], 0)
		shift = (tl+br)/2.0 - 0.5
		rms_bbs[:, :2] -= shift 
		rms_bbs[:, 2:] -= shift
		fp_eds[:, :2] -= shift
		fp_eds[:, 2:] -= shift 
		tl -= shift
		br -= shift
		eds_to_rms_tmp=[]
		for l in range(len(eds_to_rms)):
			eds_to_rms_tmp.append([eds_to_rms[l][0]])
		return rms_type,fp_eds,rms_bbs,eds_to_rms,eds_to_rms_tmp
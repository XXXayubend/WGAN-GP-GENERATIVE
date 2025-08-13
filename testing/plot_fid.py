import matplotlib.pyplot as plt
import numpy as np
import re

def extract_fid_scores(file_path):
    global_scores = {}
    epochs_order = []

    with open(file_path, 'r') as file:
        next(file)  # Sauter l'en-tête
        
        for line in file:
            if not line.strip():
                continue
                
            parts = line.split('\t')
            if len(parts) < 4:
                parts = re.split(r'\s{2,}', line.strip())
            
            if len(parts) >= 4:
                try:
                    score_type = parts[0].lower()
                    epoch = parts[1]
                    score = float(parts[2])
                    
                    if epoch not in epochs_order:
                        epochs_order.append(epoch)
                    
                    if score_type == 'global':
                        global_scores[epoch] = score
                        
                except (ValueError, IndexError) as e:
                    print(f"Erreur de traitement de ligne: {line.strip()} - {str(e)}")
    
    # RETOURNER LES VALEURS (correction cruciale)
    return [], [global_scores.get(epoch) for epoch in epochs_order], epochs_order

def plot_fid_scores(_, global_scores, epochs_labels, output_file="fid_plot.png"):
    n = len(epochs_labels)
    bar_width = 0.7  # Largeur unique
    index = np.arange(n)

    plt.figure(figsize=(14, 7), dpi=100)
    
    # Barres uniquement pour le modèle global
    bars = plt.bar(index, global_scores, 
                   width=bar_width,
                   color='blue',
                   alpha=0.8)
    
    plt.title('Score FID du Modèle Global', fontsize=16, pad=20)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Score FID', fontsize=14)
    plt.xticks(index, epochs_labels, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Ajout des valeurs
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height + 0.05, 
                 f'{height:.2f}',
                 ha='center', va='bottom',
                 fontsize=9)

    # Ajuster l'échelle Y
    plt.ylim(0, max(global_scores) * 1.25)
    plt.tight_layout()
    
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {output_file}")
    plt.show()
if __name__ == "__main__":
    INPUT_FILE = "fid_results_set8.txt"   
    OUTPUT_IMAGE = "fid_score.png"  
    
    try:
        _, global_scores, epoch_labels = extract_fid_scores(INPUT_FILE)
        
        # Filtrer les époques sans données
        valid_data = [
            (epoch, score) 
            for epoch, score in zip(epoch_labels, global_scores) 
            if score is not None
        ]
        
        if not valid_data:
            raise ValueError("Aucune donnée valide trouvée dans le fichier")
        
        epoch_labels, global_scores = zip(*valid_data)
        
        print(f"{len(epoch_labels)} époques valides :")
        for epoch, score in zip(epoch_labels, global_scores):
            print(f"Epoch {epoch}: Global={score:.2f}")
        
        plot_fid_scores(None, global_scores, epoch_labels, OUTPUT_IMAGE)
        
    except FileNotFoundError:
        print(f"ERREUR : Fichier introuvable - {INPUT_FILE}")
    except Exception as e:
        print(f"ERREUR : {str(e)}")
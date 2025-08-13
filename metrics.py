import pandas as pd
import matplotlib.pyplot as plt

# Configuration des paramètres
csv_file = 'exmps/exmps_8/train_history.csv' 
output_image = 'loss.png'  # Changement du nom de fichier pour refléter le contenu
alpha = 0.7           

# Charger les données
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Erreur: Fichier '{csv_file}' introuvable!")
    exit()

# Vérifier la présence des colonnes
required_columns = ['Epoch', 'D_loss', 'G_loss']
if not set(required_columns).issubset(df.columns):
    print("Erreur: Colonnes manquantes dans le CSV.")
    print(f"Colonnes requises: {required_columns}")
    print(f"Colonnes disponibles: {list(df.columns)}")
    exit()

# Création du plot
plt.figure(figsize=(12, 6))

# Tracer l'évolution des pertes
plt.plot(df['Epoch'], df['G_loss'], 
         color='blue', 
         alpha=alpha,
         linewidth=2,
         label='Generator Loss (G_loss)')

plt.plot(df['Epoch'], df['D_loss'], 
         color='red', 
         alpha=alpha,
         linewidth=2,
         label='Discriminator Loss (D_loss)')

# Personnalisation
plt.title('Évolution des Losses pendant l\'entraînement')
plt.xlabel('Epochs')
plt.ylabel('Valeur de Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Améliorer l'échelle des epochs
plt.xticks(range(min(df['Epoch']), max(df['Epoch'])+1, max(1, len(df)//20)))

# Sauvegarde et affichage
plt.tight_layout()
plt.savefig(output_image, dpi=300)
print(f"Graphique sauvegardé sous '{output_image}'")
plt.show()
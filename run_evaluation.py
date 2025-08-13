#!/usr/bin/env python3
import os
import sys

def main():
    print("=== Évaluation des Checkpoints ===")
    print("1. Visualisation des résultats")
    print("2. Calcul du score FID")
    print("3. Les deux")
    
    choice = input("Choisissez une option (1/2/3): ").strip()
    
    if choice == "1":
        print("\nLancement de la visualisation...")
        os.system("python visualize_checkpoints.py")
        
    elif choice == "2":
        print("\nLancement du calcul FID...")
        os.system("python evaluate_fid.py")
        
    elif choice == "3":
        print("\nLancement des deux évaluations...")
        print("1. Visualisation...")
        os.system("python visualize_checkpoints.py")
        print("\n2. FID...")
        os.system("python evaluate_fid.py")
        
    else:
        print("Option invalide")
        return
    
    print("\nÉvaluation terminée !")
    print("Résultats disponibles dans:")
    print("- visualization_results/ (images)")
    print("- fid_results.txt (scores FID)")
    print("- fid_evolution.png (graphique FID)")

if __name__ == "__main__":
    main() 
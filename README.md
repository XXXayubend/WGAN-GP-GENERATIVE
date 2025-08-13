# Generative PLAN 2D d'Architectures

Model genrative WGAN-GP to generate plan 2D  

## Structure
- `scripts/` : data preprocessing data & create model & train model
- `evaluate_Fid/` : evaluate model with score fid
- `checkponts/` : Weigth of model

## Utilisation
```bash
for train model :
python train_wgan.py
for visualate results plan geerated :
python visualize_checkpoints.py
for evaluate model :
python evaluate_fid.py


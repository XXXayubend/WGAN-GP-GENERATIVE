# Generative PLAN 2D d'Architectures

Model genrative WGAN-GP to generate plan 2D  

## Structure
- `scripts/` : data preprocessing data & create model & train model
- `evaluate_Fid/` : evaluate model with score fid
- `checkponts/` : Weigth of model

## Utilisation
```bash
python vit_train.py
python evaluate.py --model_path checkpoints/best_model.pth
python app.py 
npm start
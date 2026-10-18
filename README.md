# IFT3700 – GRP10 – Projet NHL (Étape 1)

## Installation

```bash
python -m pip install -r requirements.txt
```

## Tâche 2 – Outil de débogage interactif

L'outil permet de parcourir les événements d'une partie (saison, type de partie, partie, événement) et de voir leurs coordonnées sur une patinoire.

### Prérequis

1. Télécharger les données avec le script de la tâche 1 (à la racine du projet) :
```bash
   python download_nhl_data.py
```
   Les fichiers JSON sont placés dans `data/raw/<saison>/` (ex. `data/raw/20162017/2016020001.json`).
2. L'image de la patinoire doit être dans `figures/nhl_rink.png` (si elle est absente, une patinoire simple est dessinée à la place).

### Lancement

1. Ouvrir `tache2_interactive_tool/jupyter_notebook.ipynb` (dans VS Code, avec l'extension Jupyter, ou avec `jupyter notebook`).
2. Choisir un noyau Python où les dépendances sont installées.
3. Exécuter la cellule :
```python
   import interactive_tool
   interactive_tool.start()
```





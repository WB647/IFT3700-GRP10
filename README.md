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

## Tâche 3 – TidyData

Transforme les JSON bruts en DataFrame pandas (une ligne par tir ou but, tirs de barrage exclus).

### Lancement

Après avoir téléchargé les données (`python download_nhl_data.py`), à la racine du projet :

```bash
python tache3_tidydata.py
```

Le script construit le DataFrame de la saison 20162017, affiche les 10 premières lignes, le nombre d'événements et le nombre de valeurs manquantes par colonne.

Dans un notebook ou un autre script :

```python
from tache3_tidydata import season_to_df
df = season_to_df("20162017")
```

## Tâche 5 — Visualisations avancées : cartes de tirs

### Objectif

Cette tâche consiste à créer **cinq cartes interactives des tirs offensifs**, une pour chaque saison de 2016-17 à 2020-21. Elles permettent de sélectionner une équipe et de comparer ses tirs à la moyenne de la LNH selon leur emplacement sur la patinoire.

Le code est regroupé dans `task5_shot_maps/`, sans modifier les tâches précédentes. Les cartes et les fichiers de validation sont générés dans `task5_outputs/html/`.

### Méthode

Les données proviennent de la tâche 3 avec `game_to_df`. Les coordonnées sont ajustées pour que toutes les équipes attaquent vers la droite, en utilisant les informations des événements et le sens d'attaque.

La zone offensive est divisée en **cases de 5 × 5 pieds**. Pour chaque case, on calcule :

```text
taux_ligue = tirs_ligue / (2 × matchs_ligue)
taux_équipe = tirs_équipe / matchs_équipe
excédent = taux_équipe - taux_ligue
```

Le rouge indique un surplus de tirs par rapport à la ligue et le bleu, un déficit. La même échelle est utilisée pour les cinq saisons.

Deux affichages sont disponibles :
- **Valeurs brutes** : résultats exacts pour chaque case.
- **Carte lissée** : version plus facile à lire, avec un filtre gaussien qui conserve le total des valeurs.

### Exécution

Depuis la racine du projet :

```bash
python -m pip install -r requirements.txt
python -m task5_shot_maps.download --workers 2
python -m task5_shot_maps.generate --data-dir data/raw --output-dir task5_outputs/html
python -m task5_shot_maps.validate --data-dir data/raw --output-dir task5_outputs/html
python -m pytest tests/task5 -q
```

Les cinq fichiers HTML générés sont interactifs et fonctionnent hors ligne. Les résultats des vérifications sont enregistrés dans `generation_audit.json` et `validation_report.json`.

### Tests et validation

Les **21 tests ont réussi** et la validation des cinq saisons confirme que les cartes fonctionnent correctement.

Les vérifications portent notamment sur les coordonnées, les calculs, le lissage, les menus interactifs et la génération des fichiers.

Au total, seulement 13 coordonnées invalides ont été exclues. Les conflits de sens d'attaque ont été traités et documentés dans l'audit. Deux générations successives ont également produit des fichiers identiques.

### Limites

Les cartes montrent uniquement le **volume de tirs au but**, et non leur qualité ou leur probabilité de devenir un but. Les tirs manqués et bloqués ne sont pas inclus.

Les calculs utilisent une durée de 60 minutes par match, sans ajustement pour la prolongation. Les saisons écourtées de 2019-20 et 2020-21 peuvent aussi rendre certaines tendances moins représentatives.


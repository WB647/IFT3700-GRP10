# IFT3700 – GRP10 – Projet NHL (Étape 1)

## Installation

```bash
python -m pip install -r requirements.txt
```

## Tâche 1 — Acquisition des données

Le script `download_nhl_data.py` télécharge les données play-by-play de la LNH pour les saisons 2016–17 à 2023–24, incluant la saison régulière et les séries éliminatoires.

### Lancement

Depuis la racine du projet, exécuter :

```bash
python download_nhl_data.py
```

Le script récupère les identifiants des matchs à partir des calendriers des équipes, élimine les doublons, puis télécharge les données play-by-play de chaque match à partir de l’API de la LNH.

Les fichiers JSON sont enregistrés dans `data/raw/<saison>/`. Par exemple, les matchs de la saison 2016–17 sont enregistrés dans `data/raw/20162017/`.

Le script saute les fichiers déjà présents, ce qui permet de le relancer sans télécharger de nouveau les matchs déjà traités. Les identifiants des matchs dont le téléchargement échoue sont enregistrés dans `failed_ids.txt`, dans le dossier de la saison concernée.

Les données brutes sont exclues du dépôt Git avec `.gitignore`, car elles peuvent être recréées en exécutant le script.

IA: Pour la tâche 1, j’ai utilisé ChatGPT pour m’aider à concevoir le script de téléchargement et à organiser sa logique : récupérer les identifiants des matchs, éliminer les doublons, télécharger les fichiers JSON et reprendre le téléchargement sans retraiter les fichiers déjà présents. 


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
 IA : Pour la tache 2,demander de generer le code pour cette tache et fourni une capture de l'exemple de l'énoncé et demandé de reproduire ses dimensions et sa mise en page (patinoire de -100 à 100 pieds en x et de -42.5 à 42.5 en y, noms des équipes au-dessus, axes en « feet »). M'a egalement guide pour setup Jupyter sur VS Code

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
IA : Pour la tâche 3,Claude a proposé un code qui vérifie d'abord les vrais noms de champs. Nous avons exécuté ce script sur un vrai fichier JSON et collé la sortie dans la conversation, puis Claude a écrit le code final d'après ces champs. 


## Tâche 4 — Visualisations simples

Le script `tache4_visualizations.py` produit trois figures pour analyser les tirs et les buts dans la LNH.

### Exécution

Depuis la racine du projet, exécuter :

```bash
python tache4_visualizations.py
```

Les figures sont enregistrées dans le dossier `Figures/` :

- `Figures/task4_q1_20182019.png` : compare le nombre de tirs, le nombre de buts et le pourcentage de buts par type de tir pendant la saison 2018–19.
- `Figures/task4_q2_goal_probability_by_distance.png` : compare le pourcentage de tirs qui se terminent par un but selon la distance, pour les saisons 2018–19, 2019–20 et 2020–21.
- `Figures/task4_q3_goal_percentage_distance_and_type.png` : présente le pourcentage de buts selon la distance et le type de tir pendant la saison 2018–19.

### Résultats

Pendant la saison 2018–19, le tir du poignet (*wrist shot*) est le type de tir le plus fréquent. Toutefois, le nombre de tirs et le pourcentage de buts mesurent des aspects différents : les tirs de type *tip-in* et *deflected* ont des pourcentages de buts plus élevés dans les données analysées.

Pour les trois saisons comparées, les tirs pris près du filet ont généralement un pourcentage de buts plus élevé. Ce pourcentage diminue avec la distance. Les variations observées pour certaines distances éloignées doivent être interprétées avec prudence, car elles peuvent dépendre du nombre de tirs dans ces catégories.

IA: j’ai utilisé ChatGPT pour m’aider à écrire et à corriger le script de visualisation, notamment pour créer les graphiques sur les types de tirs, la distance et le pourcentage de buts. 


## Tâche 5 — Visualisations avancées : cartes de tirs

### Objectif

Cette tâche consiste à créer **cinq cartes interactives des tirs offensifs**, une pour chaque saison de 2016-17 à 2020-21. Elles permettent de sélectionner une équipe et de comparer ses tirs à la moyenne de la LNH selon leur emplacement sur la patinoire.

Le code est regroupé dans `task5_shot_maps/`,Les cartes et les fichiers de validation sont générés dans `task5_outputs/html/`.

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

Les cinq fichiers HTML générés sont interactifs et fonctionnent hors ligne. Les résultats des vérifications sont enregistrés dans `generation.json` et `validation_report.json`.

### Tests et validation

Les **21 tests ont réussi** et la validation des cinq saisons confirme que les cartes fonctionnent correctement.

Les vérifications portent notamment sur les coordonnées, les calculs, le lissage, les menus interactifs et la génération des fichiers.

Au total, seulement 13 coordonnées invalides ont été exclues. Les conflits de sens d'attaque ont été traités et documentés. Deux générations successives ont également produit des fichiers identiques.

### Limites

Les cartes montrent uniquement le **volume de tirs au but**, et non leur qualité ou leur probabilité de devenir un but. Les tirs manqués et bloqués ne sont pas inclus.

Les calculs utilisent une durée de 60 minutes par match, sans ajustement pour la prolongation. Les saisons écourtées de 2019-20 et 2020-21 peuvent aussi rendre certaines tendances moins représentatives.
### Utilisation de l'intelligence artificielle

L'intelligence artificielle a été utilisée comme outil d'assistance pour certaines parties de la tâche 5, notamment pour aider à la programmation et à la structuration du code des modules suivants :

- **`data.py`** : prépare et oriente les données.
- **`figure.py`** : construit les cartes interactives.
- **`generate.py`** : coordonne le pipeline et génère les fichiers.

L'IA a également été utilisée pour améliorer la mise en forme des fichiers README et corriger les fautes d'orthographe.

Le code a ensuite été compris et vérifié à l'aide des tests et des validations présentés précédemment afin de confirmer le bon fonctionnement des fonctionnalités.

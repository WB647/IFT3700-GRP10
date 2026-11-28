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

## Étape 2 - Tâche 1 - Suivi des expériences avec Weights & Biases

### Configuration

- **Équipe W&B :** `IFT3700-2026-10` 
- **Projet :** `ift3700-milestone-2`.
- **Identifiants :** jamais dans le code. Chaque membre crée un fichier `.env` à la racine (exclu par `.gitignore`) :

  ```
  WANDB_API_KEY=<votre clé>
  WANDB_ENTITY=IFT3700-2026-10
  ```

### Ce qui est suivi

| Tâche | Expériences W&B | Contenu |
|---|---|---|
| 3 | `logreg_distance`, `logreg_angle`, `logreg_distance_angle` | métriques, modèle |
| 4 | `task4-wpg-v-wsh-2017021065` | artefact `wpg_v_wsh_2017021065` (DataFrame du match 2017021065) |
| 5 | `xgb_baseline`, `xgb_tuned_full`, `xgb_selected` | configuration, métriques, figures, modèle |
| 6 | `task6_<nom-du-modèle>` (tag `task-6`) | configuration, métriques, figures, modèle |
| 7 | `task7_test_evaluation` (tag `task-7`) | modèles utilisés, métriques, 8 figures, tableau |

### Registre de modèles (`wandb-registry-model`)

| Collection | Origine |
|---|---|
| `logreg_distance`, `logreg_angle`, `logreg_distance_angle` | Tâche 3 |
| `xgb_tuned_full`, `xgb_selected` | Tâche 5 |
| `task6_hgb_v2` | Tâche 6 (modèle final) |

### Reproduire les tâches 2 à 5 avec le suivi en ligne

```bash
python Jalon2_tache2_features.py
python Jalon2_tache3_baseline.py
python -m milestone2.generate_features --dataset --data-dir data/raw --output-dir milestone2_outputs --wandb-mode online --wandb-project ift3700-milestone-2
python -m milestone2.train_xgboost --dataset milestone2_outputs/task4_features_2016_2022.csv.gz --output-dir milestone2_outputs --tuning-iterations 8 --cv 3 --wandb-mode online --wandb-project ift3700-milestone-2
```

`generate_features` crée une nouvelle version de l'artefact `wpg_v_wsh_2017021065` à chaque exécution en ligne : ne le relancer en ligne qu'une seule fois.

---


## Étape 2 — Tâches 4 et 5

Le dossier `milestone2/` ajoute les caractéristiques avancées et les expériences XGBoost demandées à l'étape 2, `milestone2.features` réutilise `tache3_tidydata.game_to_df` par l'adaptateur d'orientation existant, puis relit la liste complète `plays` afin d'associer à chaque tir l'événement immédiatement précédent.

Installation et génération de l'artefact local du match 2017021065 :

```bash
python -m pip install -r requirements.txt
python -m milestone2.generate_features --data-dir data/raw --output-dir milestone2_outputs
```

Après avoir téléchargé toutes les saisons 2016/17 à 2022/23, construire le jeu complet puis entraîner les modèles :

```bash
python -m milestone2.generate_features --dataset --data-dir data/raw --output-dir milestone2_outputs
python -m milestone2.train_xgboost --dataset milestone2_outputs/task4_features_2016_2022.csv.gz --output-dir milestone2_outputs --tuning-iterations 8 --cv 3 --wandb-mode offline
```

Pour W&B, définir `WANDB_API_KEY` et `WANDB_ENTITY`, puis ajouter `--wandb-mode online --wandb-project <projet>` aux deux commandes. Aucun secret n'est stocké dans le dépôt. Les modèles complet et sélectionné sont alors ajoutés au registre; le modèle de référence est seulement journalisé, conformément aux consignes.

Tests :

```bash
python -m pytest tests/milestone2 -q
```

L'exécution complète du 21 novembre 2026 a produit 520 564 observations. Sur la validation 2022/23 (81 990 observations), les AUC obtenues sont 0,7012 pour XGBoost distance-angle, 0,7488 pour le modèle complet réglé et 0,7428 après sélection de variables. Les quatre graphiques demandés et deux graphiques justificatifs sont sous `milestone2_outputs/figures/`. Les 9 tests propres à cette étape, ainsi que les 21 tests déjà présents, réussissent (30 tests au total). Le suivi W&B a été validé en mode hors ligne; l'authentification et la publication des liens restent à effectuer par l'équipe.

## Tâche 6 – Faites de votre mieux

### Fichiers

| Fichier | Rôle |
|---|---|
| `milestone2/evaluation.py` | Métriques (AUC, log loss, Brier, ECE, ratio xG/buts) et prédictions sauvegardées |
| `milestone2/task6_experiments.py` | Catalogue des tentatives (une fonction décorée par `@experiment`) |
| `milestone2/train_task6.py` | Programme principal (`list`, `build-dataset`, `run`, `figures`) |
| `milestone2/features.py` | Nouvelles caractéristiques (`NEW_FEATURE_COLUMNS`, `FEATURE_COLUMNS_V2`) |

### Tentatives

| Nom | Tentative | Description |
|---|---|---|
| `xgb_task5` | Référence | XGBoost réglé de la tâche 5, réutilisé tel quel (non envoyé à W&B) |
| `xgb_features_v2` | A | Hyperparamètres de la tâche 5, plus 7 nouvelles caractéristiques (filet désert, nombre de patineurs, domicile, équipe de l'événement précédent, écart au score avant le tir) |
| `xgb_tuned_v2` | B | Recherche aléatoire (log loss, validation croisée groupée par match, régularisation), valeurs manquantes gérées par XGBoost |
| `hgb_v2` | C | HistGradientBoosting (scikit-learn), catégories natives, arrêt précoce |
| `mlp_v2` | C | Réseau de neurones, normalisation par quantiles, arrêt précoce sur la log loss |
| `logreg_v2` | C | Régression logistique sur les 22 caractéristiques |
| `xgb_weighted_v2` | D | Pondération des buts (`scale_pos_weight`) |
| `xgb_isotonic_v2`, `xgb_sigmoid_v2` | D | Calibration isotonique ou sigmoïde, apprise sur 2021-22 |

### Commandes

```bash
# Lister les tentatives
python -m milestone2.train_task6 list

# Construire le jeu de données 2016-17 à 2022-23 avec les nouvelles caractéristiques (long)
python -m milestone2.train_task6 build-dataset

# Référence, puis les tentatives (une expérience W&B chacune)
python -m milestone2.train_task6 run --experiment xgb_task5
python -m milestone2.train_task6 run --experiment xgb_features_v2 xgb_tuned_v2 --wandb-mode online
python -m milestone2.train_task6 run --experiment logreg_v2 hgb_v2 mlp_v2 --wandb-mode online
python -m milestone2.train_task6 run --experiment xgb_weighted_v2 xgb_isotonic_v2 xgb_sigmoid_v2 --wandb-mode online

# Tableau récapitulatif de toutes les tentatives, puis les 4 figures du blogue
python -m milestone2.train_task6 figures
python -m milestone2.train_task6 figures --curves xgb_task5 xgb_features_v2 mlp_v2 xgb_isotonic_v2 --final hgb_v2
```

### Options de `run`

| Option | Effet |
|---|---|
| `--wandb-mode online` | Envoie l'expérience à W&B (`disabled` par défaut) |
| `--register` | Lie le modèle au registre (collection `task6_<nom>`) |
| `--param clé=valeur` | Option propre à la tentative, p. ex. `--param n_iter=20` (B) ou `--param base=xgb_tuned_v2` (D) |
| `--name <nom>` | Enregistre une variante sous un autre nom, p. ex. `--name xgb_weighted_isotonic` |

### Résultats

| Fichier | Contenu |
|---|---|
| `milestone2_outputs/task6/predictions/<nom-du-modèle>.csv.gz` | Probabilités de validation |
| `milestone2_outputs/task6/metrics/<nom-du-modèle>.json` | Métriques, paramètres, lien W&B |
| `milestone2_outputs/task6/figures/runs/` | Figures de chaque tentative (importance, réglage, courbe d'apprentissage, calibration) |
| `milestone2_outputs/task6/summary.md` | Tableau récapitulatif |
| `milestone2_outputs/figures/task6_*.png` | Les 4 figures du blogue |

### Ajouter une tentative

Ajouter une fonction dans `milestone2/task6_experiments.py` :

```python
@experiment("mon_essai", features=FEATURE_COLUMNS_V2, description="...", tags=("xgboost",))
def mon_essai(train, context):
    model = make_pipeline(context.features, {"n_estimators": 300})
    model.fit(train[context.features], train["is_goal"])
    return model, {"hyperparameters": {"n_estimators": 300}}
```

Elle est alors disponible avec `run --experiment mon_essai`.

---

## Tâche 7 – Évaluation sur l'ensemble de test

### Commande

```bash
python -m milestone2.evaluate_test
```

| Option | Effet |
|---|---|
| `--alias <version>` | Version prise dans le registre (`latest` par défaut) |
| `--registry <chemin>` | Préfixe du registre, si W&B demande le nom de l'organisation |
| `--bootstrap <n>` | Nombre de tirages pour les intervalles de confiance (500 par défaut, 0 pour désactiver) |
| `--source local --wandb-mode disabled` | Essai avec les modèles du disque, sans W&B |

### Résultats

| Fichier | Contenu |
|---|---|
| `milestone2_outputs/figures/task7_regular_*.png` | 4 figures, saison régulière (5 courbes chacune) |
| `milestone2_outputs/figures/task7_playoffs_*.png` | 4 figures, séries |
| `milestone2_outputs/task7/summary.md` | Données, métriques avec intervalles de confiance à 95 % (bootstrap sur les matchs), écarts avec le modèle final, comparaison validation / test |
| `milestone2_outputs/task7/metrics.csv` | Toutes les métriques |
| `milestone2_outputs/task7/predictions/` | Probabilités de chaque modèle (validation, saison régulière, séries) |
| `milestone2_outputs/task7/run.json` | Versions des modèles évalués et lien W&B |

---

## Fichiers exclus du dépôt

Ces fichiers sont volumineux ou recalculables (voir `.gitignore`) :

```
.env
milestone2_outputs/task6_features_*.csv.gz
milestone2_outputs/task6/models/
milestone2_outputs/task7/test_features_*.csv.gz
milestone2_outputs/task7/downloaded_models/
```

Les modèles eux-mêmes sont conservés dans W&B.

## Déclaration d'utilisation de l'IA

Une grande partie du code des tâches 6 et 7 et leurs tests ont été écrits avec l'aide de Claude (Anthropic), plus précisément avec le modèle Opus 5.5 :
- `milestone2/evaluation.py`, `milestone2/task6_experiments.py`, `milestone2/train_task6.py`, `milestone2/evaluate_test.py` ;
- `tests/milestone2/test_task6.py`, `tests/milestone2/test_evaluate_test.py` ;
- les modifications de `features.py`, `models.py`, `wandb_tracking.py` et `Jalon2_tache2_features.py` (séries, prolongations, nouvelles caractéristiques).

L'équipe a relu, réédité, exécuté et validé ce code et reste responsable de son contenu.
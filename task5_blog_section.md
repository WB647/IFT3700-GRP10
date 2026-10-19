
## Tâche 1 — Acquisition des données play-by-play

Pour analyser les matchs de la LNH, nous avions besoin des données play-by-play de chaque rencontre. Elles décrivent les événements d’un match, notamment les tirs, les buts, les mises au jeu et les arrêts.

Nous avons téléchargé les saisons 2016–17 à 2023–24, en incluant la saison régulière et les séries éliminatoires.

### Source des données et sélection des matchs

Nous avons utilisé l’API publique de la LNH. Pour chaque saison, le script consulte le calendrier des équipes avec l’endpoint `club-schedule-season`, récupère les identifiants des matchs de saison régulière et des séries, puis élimine les doublons avec un ensemble Python (`set`).

Le script télécharge ensuite le play-by-play de chaque match à partir de l’endpoint suivant :

```text
https://api-web.nhle.com/v1/gamecenter/{GAME_ID}/play-by-play
```

Cette méthode évite de construire manuellement les identifiants des matchs. Un match qui apparaît dans les calendriers des deux équipes n’est téléchargé qu’une seule fois.
Installation et exécution
Depuis la racine du projet, installez les dépendances :
```bash
python -m pip install -r requirements.txt`
```
Lancez ensuite le téléchargement :
```bash
python download_nhl_data.py
```
Par exemple, les fichiers de la saison 2016–17 se trouvent dans `data/raw/20162017/`.

Avant de télécharger un match, le script vérifie si son fichier JSON existe déjà. Si c’est le cas, il le saute. On peut donc relancer le script pour reprendre le téléchargement sans télécharger de nouveau les matchs déjà enregistrés.

Si le téléchargement d’un match échoue, son identifiant est ajouté à un fichier failed_ids.txt dans le dossier de la saison concernée. Le script affiche également le nombre de matchs trouvés, la progression tous les 100 matchs et un message de fin pour chaque saison.
###Vérification des données
Après l’exécution, nous avons vérifié le nombre de fichiers JSON dans chaque dossier de saison. Les nombres obtenus correspondaient aux nombres de matchs trouvés par le script :
| Saison | Fichiers JSON |
|---|---:|
| 2016–17 | 1317 |
| 2017–18 | 1355 |
| 2018–19	| 1358 |
| 2019–20 | 1212 |
| 2020–21 | 952 |    
| 2021–22	| 1401 |      
| 2022–23	| 1400 |        
| 2023–24	| 1400 |        

Aucun fichier failed_ids.txt n’a été trouvé après le téléchargement.

Les données brutes ne sont pas incluses dans le dépôt Git. Elles sont volumineuses et peuvent être recréées en exécutant `download_nhl_data.py`. Le dépôt contient donc le script nécessaire pour que les membres de l’équipe puissent télécharger leur propre copie des données.
Résultat
Nous avons obtenu un fichier play-by-play par match pour les huit saisons étudiées. Ces données brutes servent de source aux étapes suivantes du projet : exploration interactive, création d’un DataFrame ordonné et visualisation des tirs.

## Tâche 2 — Outil de débogage interactif

Nous avons créé un outil interactif avec ipywidgets et matplotlib. Il permet de choisir une saison, de basculer entre la saison régulière et les séries, puis de parcourir les parties et leurs événements avec des curseurs. Les coordonnées de l'événement sont dessinées sur la patinoire (en pieds) et ses données sont affichées, même quand il n'y a pas de coordonnées.

<figure>
  <img src="{{ '/figures/task2_interactive_tool.png' | relative_url }}" alt="Capture de l'outil interactif" style="max-width:100%;">
  <figcaption>Figure 1 — Capture de l'outil interactif (saison 2016-2017).</figcaption>
</figure>

<details>
<summary>Code de l'outil (<code>interactive_tool.py</code>)</summary>


```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
from ipywidgets import IntSlider, interact
from matplotlib.patches import Circle

# Paths are relative to the repository root, whatever folder the notebook is run from.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
RINK_IMAGE = ROOT / "figures" / "nhl_rink.png"  # rink image (200 ft x 85 ft)


def get_seasons():
    """Return the season folders found in the data directory (e.g. 20162017)."""
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit())


def get_games(season, game_type):
    """Return the JSON files of one season. game_type: '02' regular season, '03' playoffs."""
    return sorted((DATA_DIR / season).glob(f"{season[:4]}{game_type}*.json"))


def draw_rink(ax):
    """Draw the rink (x from -100 to 100 ft, y from -42.5 to 42.5 ft)."""
    if RINK_IMAGE.exists():
        ax.imshow(plt.imread(RINK_IMAGE), extent=[-100, 100, -42.5, 42.5])
    else:
        ax.plot([-100, 100, 100, -100, -100], [-42.5, -42.5, 42.5, 42.5, -42.5], "k")
        ax.axvline(0, color="red")  # center line
        ax.axvline(-25, color="blue")  # blue lines
        ax.axvline(25, color="blue")
        ax.axvline(-89, color="red", linewidth=1)  # goal lines
        ax.axvline(89, color="red", linewidth=1)
        ax.add_patch(Circle((0, 0), 15, fill=False, color="blue"))  # center circle

    ax.set_xlim(-100, 100)
    ax.set_ylim(-42.5, 42.5)
    ax.set_xticks(range(-100, 101, 25))
    ax.set_yticks([-42.5, -21.25, 0, 21.25, 42.5])
    ax.set_xlabel("feet")
    ax.set_ylabel("feet")


def show_event(season, game_type, game_index, event_index):
    """Print the details of one event and draw its coordinates on the rink."""
    games = get_games(season, game_type)
    if not games:
        print("No game found for this selection.")
        return

    # The sliders have a fixed maximum, so stay inside the real number of games/events.
    game_index = min(game_index, len(games) - 1)
    game = json.loads(games[game_index].read_text(encoding="utf-8"))

    plays = game.get("plays", [])
    if not plays:
        print("No event in this game.")
        return
    event_index = min(event_index, len(plays) - 1)
    play = plays[event_index]
    details = play.get("details", {})

    home, away = game["homeTeam"]["abbrev"], game["awayTeam"]["abbrev"]
    period = play["periodDescriptor"]["number"]
    print(game["gameDate"])
    print(f"Game ID: {game['id']}; {home} (home) vs {away} (away)  [{game_index + 1}/{len(games)}]")
    print(f"Event {event_index + 1}/{len(plays)}: {play['typeDescKey']}, "
          f"period {period}, {play['timeInPeriod']}")
    print(details)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    draw_rink(ax)

    # Team names above the rink, on the side they defend (left/right of the rink).
    left, right = (home, away) if play.get("homeTeamDefendingSide") == "right" else (away, home)
    ax.text(-50, 44, left, ha="center", va="bottom")
    ax.text(50, 44, right, ha="center", va="bottom")

    title = f"{play['typeDescKey']}\n{play['timeInPeriod']} P-{period}"
    if "xCoord" in details and "yCoord" in details:
        ax.scatter(details["xCoord"], details["yCoord"], s=120, color="blue")
    else:
        title += " (no coordinates)"
    ax.set_title(title, y=1.1)
    plt.show()


def start():
    """Display the interactive tool (call it from a notebook cell)."""
    seasons = get_seasons()
    if not seasons:
        print(f"No season found in {DATA_DIR}. Run download_nhl_data.py first.")
        return

    interact(
        show_event,
        season=seasons,
        game_type=[("Saison régulière", "02"), ("Séries", "03")],
        game_index=IntSlider(min=0, max=1400),
        event_index=IntSlider(min=0, max=600),
    )
```
{% endraw %}

</details>

## Tâche 3 — Données ordonnées

Nous avons transformé les fichiers JSON bruts en un DataFrame pandas, avec une ligne par tir ou but (les tirs de barrage sont exclus). Pour la saison 2016-2017, le DataFrame contient 79 798 événements : 74 362 en saison régulière et 5 436 en séries. Chaque événement est identifié de façon unique par `game_id` et `event_id`, et contient la saison, le type de match, la période, le temps, l'équipe qui tire, un indicateur but ou tir, les coordonnées, le tireur, le gardien, le type de tir, un indicateur de filet désert et la force sur la glace. Le code est dans `tache3_tidydata.py`.

### 1. Extrait du DataFrame

<figure>
  <img src="{{ '/figures/task3_head10.png' | relative_url }}" alt="Extrait du DataFrame" style="max-width:100%;">
  <figcaption>Figure 2 — Les 10 premières lignes du DataFrame (<code>df.head(10)</code>).</figcaption>
</figure>

Valeurs manquantes. Elles sont gardées comme manquantes (pas de 0 ni de faux) et exclues seulement des analyses qui en ont besoin. Pour 2016-2017, 4 événements n'ont pas de coordonnées, 6 n'ont pas de type de tir et 314 n'ont pas de gardien. Ces 314 cas correspondent à des filets déserts : l'absence de gardien n'arrive jamais ailleurs.

Force sur la glace. Le `situationCode` s'écrit `[gardien visiteur][patineurs visiteurs][patineurs locaux][gardien local]`. Par exemple, `1451` veut dire que le visiteur a 4 patineurs et le local 5. Pour chaque tir ou but, on prend le nombre de patineurs de l'équipe qui tire et celui de l'équipe adverse, en inversant selon que l'équipe qui tire est locale ou visiteuse. Si les deux nombres sont égaux, c'est de l'égalité ; si l'équipe qui tire en a plus, c'est un avantage numérique ; sinon, un désavantage numérique. La force exacte (`5v4`, `5v3`, etc.) est simplement ces deux nombres. Un chiffre de gardien égal à 0 pour l'équipe qui défend indique un filet désert.

Cas limites.
- Un gardien retiré pour un attaquant de plus donne un 6 contre 5. Par nombre de patineurs, ce cas est compté comme un avantage numérique, alors qu'il n'y a pas de pénalité. On peut le repérer, car le chiffre du gardien de l'équipe vaut 0.
- La prolongation à 3 contre 3 est comptée comme de l'égalité, même si ce n'est pas du 5 contre 5. Le champ `strength_state` garde ce détail.
- Une pénalité différée n'est pas visible dans le code : le jeu continue à forces égales jusqu'à ce que l'équipe fautive touche la rondelle.
- Sans `situationCode`, on pourrait reconstruire la force avec les événements de pénalité (équipe, durée, temps de début) : à chaque pénalité, on retire un patineur à l'équipe fautive jusqu'à la fin de la pénalité ou jusqu'à un but en avantage numérique pour une pénalité mineure.

### 2. Caractéristiques supplémentaires

Voici trois caractéristiques qu'on pourrait créer à partir des données :
- Distance et angle du tir. Avec les coordonnées `x` et `y` et la position du filet, on peut calculer la distance au filet et l'angle de tir. Ce sont des caractéristiques liées à la probabilité de marquer.
- Rebond. Un tir peut être classé comme rebond s'il vient de la même équipe, dans la même période, moins de 3 secondes après un autre tir de cette équipe. On l'identifie en comparant le temps et l'équipe de chaque tir avec ceux du tir précédent de la même partie.
- Tir en contre-attaque. Un tir peut être classé comme contre-attaque s'il vient peu de temps après un événement dans l'autre moitié de la patinoire, c'est-à-dire si la rondelle a parcouru une grande distance en `x` en quelques secondes. On l'identifie avec le temps et les coordonnées de l'événement précédent.
=======


## Tâche 4 — Visualisations simples des tirs et des buts

Cette tâche présente trois visualisations qui explorent les tirs et les buts dans les données de la LNH. Elles portent sur les types de tirs, la distance par rapport au filet et le pourcentage de tirs qui se terminent par un but.

### 1. Nombre de tirs et pourcentage de buts selon le type de tir

La première figure compare, pour la saison 2018–19, le nombre de tirs, le nombre de buts et le pourcentage de tirs qui se terminent par un but pour chaque type de tir.

![Nombre de tirs, nombre de buts et pourcentage de buts par type de tir, saison 2018-19]({{ '/Figures/task4_q1_20182019.png' | relative_url }})

*Figure 1 — Tirs et buts par type de tir pendant la saison 2018–19.*

Le tir du poignet (*wrist shot*) est le type de tir le plus fréquent dans les données. Cependant, le type de tir le plus fréquent n’a pas nécessairement le pourcentage de buts le plus élevé. Les tirs de type *tip-in* et *deflected* affichent des pourcentages de buts plus élevés dans les données analysées. Ces résultats doivent être interprétés en tenant compte du nombre de tirs observés pour chaque catégorie.

### 2. Pourcentage de buts selon la distance

La deuxième figure compare le pourcentage de tirs qui se terminent par un but selon la distance, pour les saisons 2018–19, 2019–20 et 2020–21.

![Pourcentage de buts selon la distance pour trois saisons]({{ '/Figures/task4_q2_goal_probability_by_distance.png' | relative_url }})

*Figure 2 — Pourcentage de tirs qui se terminent par un but selon la distance.*

Les tirs pris près du filet ont généralement un pourcentage de buts plus élevé. Ce pourcentage tend à diminuer lorsque la distance augmente. Les variations observées dans les catégories les plus éloignées doivent être interprétées avec prudence, car elles peuvent être influencées par le nombre de tirs dans chaque catégorie.

### 3. Pourcentage de buts selon la distance et le type de tir

La troisième figure présente le pourcentage de buts selon la distance et le type de tir pour la saison 2018–19.

![Pourcentage de buts selon la distance et le type de tir, saison 2018-19]({{ '/Figures/task4_q3_goal_percentage_distance_and_type.png' | relative_url }})

*Figure 3 — Pourcentage de buts selon la distance et le type de tir pendant la saison 2018–19.*

Cette figure combine les deux dimensions précédentes. Elle permet de comparer les types de tirs à différentes distances et de voir que le pourcentage de buts varie selon l’emplacement et le type de tir. Les catégories qui contiennent peu de tirs peuvent produire des pourcentages plus variables.

### Exécution et reproductibilité

Depuis la racine du projet, exécutez :

```bash
python tache4_visualizations.py
```
Le script génère les trois figures dans le dossier Figures/. Les résultats présentés dans cette section proviennent des données play-by-play téléchargées pour les tâches précédentes.
>>>>>>> 5e55510403f71f41eb4e1e249ea118c9367abbbf




## Tâche 5 — Cartes de tirs offensives

Les cinq cartes montrent où chaque équipe tire plus ou moins souvent que la moyenne de la LNH, dans des cases de 5 × 5 pieds. Le rouge représente un surplus de tirs et le bleu, un déficit. Toutes les équipes attaquent vers la droite et la même échelle est utilisée pour faciliter les comparaisons entre les saisons.

Le menu **Affichage** permet de passer des données brutes à une version lissée, qui rend les tendances plus faciles à voir. Le lissage répartit les valeurs entre les cases voisines sans modifier le total des tirs. Les analyses ci-dessous utilisent uniquement les données brutes.

### 1. Les cinq cartes interactives

<figure>
  <iframe src="{{ '/task5_outputs/html/shot_map_20162017.html' | relative_url }}" width="100%" height="650" loading="lazy" title="Carte de tirs 2016-2017" style="border:0;"></iframe>
  <figcaption>Figure 1 — Saison 2016-17 (30 équipes).</figcaption>
</figure>

<figure>
  <iframe src="{{ '/task5_outputs/html/shot_map_20172018.html' | relative_url }}" width="100%" height="650" loading="lazy" title="Carte de tirs 2017-2018" style="border:0;"></iframe>
  <figcaption>Figure 2 — Saison 2017-18 (31 équipes avec l'arrivée de Vegas).</figcaption>
</figure>

<figure>
  <iframe src="{{ '/task5_outputs/html/shot_map_20182019.html' | relative_url }}" width="100%" height="650" loading="lazy" title="Carte de tirs 2018-2019" style="border:0;"></iframe>
  <figcaption>Figure 3 — Saison 2018-19.</figcaption>
</figure>

<figure>
  <iframe src="{{ '/task5_outputs/html/shot_map_20192020.html' | relative_url }}" width="100%" height="650" loading="lazy" title="Carte de tirs 2019-2020" style="border:0;"></iframe>
  <figcaption>Figure 4 — Saison 2019-20 (écourtée, avec des taux ajustés au nombre de matchs).</figcaption>
</figure>

<figure>
  <iframe src="{{ '/task5_outputs/html/shot_map_20202021.html' | relative_url }}" width="100%" height="650" loading="lazy" title="Carte de tirs 2020-2021" style="border:0;"></iframe>
  <figcaption>Figure 5 — Saison 2020-21 (56 matchs par équipe).</figcaption>
</figure>

Au total, les cartes couvrent 5 722 matchs. Seulement 13 tirs ont été exclus en raison de coordonnées invalides. Les tirs situés hors de la zone affichée sont comptabilisés séparément.

### 2. Comment interpréter les cartes

Les cartes permettent de voir où les équipes concentrent leurs tirs : près du filet, au centre (slot), sur les côtés ou à la pointe. Elles permettent aussi de comparer les habitudes offensives d'une saison à l'autre.

Par contre, **tirer davantage ne signifie pas nécessairement mieux attaquer**. Les cartes montrent le volume de tirs au but, mais pas leur qualité ni leurs chances de devenir des buts. Elles ne tiennent pas non plus compte des tirs bloqués, du contexte de jeu ou de la performance des gardiens.

### 3. Colorado : 2016-17 contre 2020-21

L'évolution du Colorado est assez frappante. En 2016-17, l'équipe enregistrait **2,24 tirs de moins par match** que la moyenne de la ligue dans la zone affichée, avec un déficit particulièrement marqué au centre de la zone offensive.

En 2020-21, la situation s'inverse : Colorado produit **4,90 tirs de plus par match** que la moyenne. Les zones centrales, auparavant déficitaires, deviennent également excédentaires.

Cette progression correspond aux résultats de l'équipe. Colorado est passé du dernier rang de la LNH en 2016-17 (48 points) au premier rang en 2020-21 (82 points), selon les [statistiques officielles de la LNH](https://records.nhl.com/col/overview/season-by-season-record).

Les cartes montrent donc une attaque beaucoup plus présente offensivement, notamment dans les zones centrales. Elles ne suffisent toutefois pas à expliquer à elles seules l'amélioration des résultats.

### 4. Buffalo contre Tampa Bay (2018-19 à 2020-21)

Pour comparer les deux équipes, on s'intéresse surtout au **slot intérieur**, soit la zone centrale devant le filet (x de 55 à 75 pieds et |y| ≤ 17,5).

| Saison | Équipe | Excédent total | Excédent dans le slot | Part des tirs dans le slot |
|---|---|---:|---:|---:|
| 2018-19 | Buffalo | +1,39 | −0,20 | 19,0 % |
| 2018-19 | Tampa Bay | +0,87 | +2,70 | 28,8 % |
| 2019-20 | Buffalo | −1,91 | −0,78 | 19,3 % |
| 2019-20 | Tampa Bay | −0,05 | +2,46 | 29,0 % |
| 2020-21 | Buffalo | −1,42 | −1,34 | 16,4 % |
| 2020-21 | Tampa Bay | +0,56 | +0,91 | 23,0 % |

*Les excédents sont exprimés en tirs par match par rapport à la moyenne de la ligue.*

La principale différence entre les deux équipes n'est pas le nombre total de tirs, mais plutôt **l'endroit d'où elles tirent**.

Tampa Bay affiche un surplus de tirs dans le slot durant les trois saisons, tandis que Buffalo reste toujours sous la moyenne. C'est particulièrement intéressant en 2018-19 : Buffalo tire davantage que Tampa au total, mais beaucoup moins dans les zones centrales.

Cette tendance accompagne les résultats des équipes. Tampa Bay remporte la Coupe Stanley en 2020 et 2021, alors que Buffalo connaît plusieurs saisons difficiles ([statistiques de Tampa Bay](https://records.nhl.com/tbl/overview/season-by-season-record), [bilan de Buffalo](https://www.nhl.com/sabres/news/nhl-nhlpa-stanley-cup-playoffs-announcement-buffalo-sabres-2019-20-sea-317020816)).

On peut donc penser que la capacité de Tampa à générer plus de tirs au centre contribue à son succès. Il faut toutefois rester prudent : la qualité des tirs, la défense et les gardiens jouent aussi un rôle important.

### Méthode et reproductibilité

Pour chaque case, on calcule le nombre de tirs par match d'une équipe, puis on soustrait la moyenne de la ligue. Un résultat positif indique un surplus de tirs et un résultat négatif, un déficit.

Les coordonnées ont été réorientées pour que toutes les équipes attaquent vers la droite. La version lissée utilise un filtre gaussien léger (σ = 3 pieds) qui conserve le total des valeurs.

Les détails du traitement des données, les tests et les vérifications sont disponibles dans le README et dans `task5_outputs/html/generation_audit.json`.

**Références techniques :**  [API de la LNH](https://api-web.nhle.com/v1/gamecenter/2023021190/play-by-play) · [Documentation Plotly](https://plotly.com/python/interactive-html-export/)

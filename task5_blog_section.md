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
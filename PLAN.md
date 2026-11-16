# PLAN.md : Pacing Optimizer v2, sur données réelles

Ce fichier sert de cahier des charges pour Claude Code. Lis-le en entier avant de commencer, puis lis `pacing_model.py`, `app.py`, `examples/tdf2026_stage16.py` et `README.md`.

## 1. Contexte

Je suis Vadim Digonnet, data scientist freelance spécialisé dans le sport d'endurance. Ce repo est ma vitrine auprès d'équipes cyclistes UCI et de triathlètes pro. Je veux un outil qui tienne la route devant le staff performance d'une équipe : données réelles, hypothèses écrites, erreurs mesurées.

## 2. État actuel

- `pacing_model.py` : modèle de puissance de Martin et al. (1998), vitesse d'équilibre par segment, bilan W' de Skiba, optimisation SLSQP du temps sous contrainte W' ≥ 0. `optimize_pacing` accepte un paramètre `bounds` pour figer la puissance sur certains segments.
- `app.py` : dashboard Streamlit avec curseurs athlète, tableau de parcours éditable et graphiques matplotlib.
- `examples/tdf2026_stage16.py` : rejoue le CLM Évian-Thonon du Tour 2026 à partir d'un profil reconstruit à la main (6 segments). Il estime la puissance d'Evenepoel par secteur à partir de ses pointages, cale CP, puis optimise. Résultat actuel : environ 22 s de gain potentiel.

Limites connues :
- Le parcours est approximatif : pas de GPX, pente de montée supposée uniforme.
- Le vent est ignoré et la densité de l'air est fixée à la main.
- Le modèle suppose une vitesse d'équilibre par segment : pas d'inertie, pas de virages, pas de freinage. C'est pour ça que la descente est figée dans l'exemple.
- Masse, CdA et W' sont des hypothèses, et le calage ne porte que sur un seul coureur.
- Aucune validation chiffrée de la précision du modèle.

## 3. Objectif

Passer d'une démo à un outil validé :
1. Parcours issus de vrais GPX.
2. Météo réelle (vent projeté sur le cap, densité de l'air calculée).
3. Physique dynamique avec virages et freinage, pour que les descentes soient modélisées au lieu d'être figées.
4. Calage et validation sur les temps intermédiaires réels de plusieurs coureurs et de plusieurs CLM, avec une erreur chiffrée.
5. App Streamlit v2 : import GPX, carte, vent, plan de puissance exportable.

## 4. Règles de travail

- Fais des modifications ciblées. Ne réécris pas un fichier entier quand quelques fonctions suffisent.
- Si une modification casse ou remplace un comportement qui fonctionnait, dis-le explicitement. L'API actuelle (`Rider`, `Environment`, `Course`, `simulate`, `optimize_pacing`) doit continuer à marcher : ajoute du neuf à côté au lieu de casser l'existant.
- N'invente aucune donnée. Chaque chiffre réel (temps, distance, météo, masse d'un coureur) doit venir d'une source notée dans le code ou dans un CSV, avec une colonne `source_url`. Si une donnée manque, demande-moi au lieu de supposer.
- Sépare clairement les hypothèses (CdA, masse, Crr, W') des données mesurées, dans le code comme dans les rapports.
- Respecte les conditions d'utilisation des sites. Pour ProCyclingStats ou un autre site scrapé : requêtes espacées, cache local, pas de scraping massif. Si un site l'interdit, demande-moi.
- Vérifie la licence de GoldenCheetah OpenData avant toute publication de résultats qui en dérivent.
- Mets les données brutes téléchargées en cache dans `data/raw/` et ne les modifie jamais. Les données nettoyées vont dans `data/processed/`.
- Le code et les commentaires peuvent être en français. Le README reste en anglais, car il vise des équipes étrangères.
- Avance phase par phase. À la fin de chaque phase, résume ce qui a été fait, les chiffres obtenus et ce qui reste incertain, puis propose un commit.

## 5. Sources de données

| Besoin | Source | Remarques |
|---|---|---|
| Trace GPX du CLM du Tour 2026 (étape 16) | https://cdn.cyclingstage.com/images/tour-de-france/2026/stage-16-parcours.gpx | Contrôler la distance (26,1 km) et le dénivelé positif (environ 500 m) |
| Profil de référence de l'étape 16 | https://www.intervals.icu/tour-de-france-2026/stage-16/ | Côte de Larringes : 9,54 km à 4,4 %, +403 m, sommet au km 9,64 |
| Résultats et pointages de l'étape 16 | https://www.cyclingstage.com/remo-evenepoel/stage-16-results-tdf-2026 et https://www.procyclingstats.com/race/tour-de-france/2026/stage-16/result/result | Pointages aux km 4,8, 9,7 et 18,1. Evenepoel : 8'08", 15'42", 23'30", 32'19" |
| Météo historique | API Open-Meteo archive (`archive-api.open-meteo.com/v1/archive`) | Horaire : température, pression de surface, humidité, vent à 10 m (vitesse et direction). Ramener le vent à hauteur de cycliste (environ 1 m) avec un profil logarithmique, coefficient documenté |
| Altitude de contrôle | API Open-Meteo elevation, ou API altimétrique IGN pour la France | Pour corriger un GPX dont l'altitude est bruitée |
| Horaires de départ des coureurs | Pages « start times » de l'étape (par exemple bikeradar, domestiquecycling) | Pour associer chaque coureur à sa météo |
| Données de puissance réelles | GoldenCheetah OpenData (projet OSF 10.17605/OSF.IO/6HFPZ, package Python `cheetahpy`) | Sert à fixer des plages plausibles de CP et W' et à tester le modèle W'bal, pas à représenter des pros |
| Autres CLM pour la validation | À identifier : CLM individuels récents des grands tours avec un GPX public et des pointages publiés | Propose-moi une liste de 3 à 5 courses avec leurs sources avant de les intégrer |

## 6. Phases

### Phase 0 : base saine
- Environnement virtuel, `requirements.txt` à jour, `pytest`.
- Tests de non-régression sur l'existant : la démo de 10 km doit donner un gain d'environ 44 s, et l'exemple de l'étape 16 environ 22 s. Tolérance de quelques dixièmes de seconde selon la version de SciPy.
- Critère : `pytest` passe.

### Phase 1 : parcours GPX
- Nouveau module `course_io.py` : lecture GPX (avec `gpxpy`), distance cumulée en géodésique, lissage de l'altitude (filtre Savitzky-Golay ou équivalent, paramètres documentés), rééchantillonnage tous les 50 à 100 m.
- Pour chaque segment : pente, cap, rayon de courbure.
- Critères : distance à ±1 % de 26,1 km ; dénivelé positif cohérent avec les environ 500 m annoncés ; pente moyenne de la Côte de Larringes proche de 4,4 %. Produis un graphique profil GPX contre profil publié.

### Phase 2 : météo réelle
- Nouveau module `weather.py` : appel Open-Meteo avec cache JSON dans `data/raw/`.
- Densité de l'air calculée à partir de la température, de la pression et de l'humidité, corrigée de l'altitude de chaque segment.
- Vent de face ou de dos par segment, à partir de la direction du vent et du cap.
- `Environment` doit accepter des valeurs par segment, sans casser l'usage scalaire actuel.
- Critère : un tableau par segment (rho, composante de vent) et un graphique.

### Phase 3 : physique dynamique
- Nouveau simulateur qui intègre l'équation du mouvement avec l'inertie (énergie cinétique, masse effective des roues), au lieu de la vitesse d'équilibre.
- Vitesse maximale en virage : v_max = sqrt(a_lat × r), avec une accélération latérale limite paramétrable (indique la valeur choisie et pourquoi). Freinage avec une décélération maximale paramétrable.
- `simulate` reste disponible. Le nouveau simulateur vit à côté (par exemple `simulate_dynamic`).
- Critère : sur un parcours plat sans virage, les deux simulateurs donnent le même temps à moins de 0,5 % près.

### Phase 4 : calage et validation sur l'étape 16
- Construis `data/processed/tdf2026_stage16_splits.csv` : coureur, pointage, distance, temps cumulé, heure de départ, `source_url`. Prends au moins le top 20.
- Masse des coureurs : uniquement des valeurs publiées, avec leur source. Sinon, laisse vide et demande-moi.
- Pour chaque coureur, estime la puissance par secteur à partir de ses pointages, avec sa propre météo.
- Calage : CdA partagé ou par profil morphologique, CP par coureur. Explique la méthode.
- Validation : cale le modèle sur une partie des pointages et prédis les autres (validation croisée par pointage omis). Rapporte l'erreur moyenne absolue en secondes et en pourcentage du temps de secteur.
- Critère : un tableau d'erreurs par coureur et par secteur. Pas d'objectif chiffré imposé, mais explique tout écart supérieur à 2 %.

### Phase 5 : physiologie
- Avec GoldenCheetah OpenData (après vérification de la licence), extrais des courbes puissance-durée réelles, ajuste CP et W', et compare le modèle W'bal de Skiba aux efforts observés.
- Résultat attendu : des plages réalistes de CP/kg et de W' à utiliser comme bornes et a priori dans la phase 4.

### Phase 6 : optimiseur v2
- Optimisation sur le simulateur dynamique, à une résolution de 100 à 200 m.
- Pénalité de lissage pour que le plan reste réalisable sur le vélo, et puissance regroupée par blocs lisibles (par exemple par kilomètre ou par section du parcours).
- Contrainte W'bal ≥ 0. Puissance libre en descente, puisque les virages sont maintenant modélisés.
- Critère : moins d'une minute de calcul pour un CLM de 30 km. Comparaison stratégie réelle, puissance constante et stratégie optimale pour le top 10 de l'étape 16.

### Phase 7 : validation multi-courses
- Applique tout le pipeline à 3 à 5 autres CLM (liste à valider avec moi).
- Critère : un rapport d'erreur par course. On doit voir si la précision tient hors de l'étape 16.

### Phase 8 : app Streamlit v2
- Import d'un GPX, sélection de la date et de l'heure pour la météo, profil athlète (saisie manuelle ou fichier de puissance).
- Carte du parcours avec la puissance cible par section et des flèches de vent (`pydeck` ou `folium`).
- Comparaison des stratégies, bilan W', temps par section.
- Export du plan en CSV. Export en séance pour compteur (.fit ou .zwo) si c'est faisable proprement.
- Mode démo pré-chargé avec l'étape 16 du Tour 2026.

### Phase 9 : présentation
- README en anglais mis à jour : méthode, sources, hypothèses, résultats de validation, limites.
- Un notebook ou un rapport HTML `reports/tdf2026_stage16.html`, avec graphiques soignés et lisibles par un directeur sportif.
- Prépare le déploiement sur Streamlit Community Cloud, mais ne publie rien sans mon accord.

## 7. Structure cible du repo

```
pacing-optimizer/
├── pacing_model.py        # existant, API conservée
├── course_io.py           # GPX, lissage, cap, courbure
├── weather.py             # Open-Meteo, densité, vent par segment
├── dynamics.py            # simulateur dynamique, virages, freinage
├── calibration.py         # inversion des pointages, calage, validation
├── optimizer.py           # optimiseur v2
├── app.py                 # Streamlit v2
├── examples/
├── data/raw/              # brut, en cache, jamais modifié
├── data/processed/        # nettoyé, avec sources
├── reports/
├── tests/
├── PLAN.md
└── README.md
```

Cette structure est indicative. Si tu en vois une meilleure, propose-la avant de réorganiser.

## 8. À ne pas faire

- Présenter une hypothèse comme une mesure.
- Afficher des watts « réels » de coureurs pros : on n'a que des estimations tirées des temps.
- Supprimer ou réécrire en bloc le code existant sans me prévenir.
- Committer des données dont la licence interdit la redistribution.
- Pousser sur GitHub sans me montrer le diff.

## 9. Avant de commencer

Pose-moi tes questions si l'un de ces points bloque :
- une source n'est pas accessible ou ses conditions d'utilisation posent problème ;
- un choix de modélisation a un impact fort sur les résultats (CdA par coureur ou partagé, a_lat limite, modèle de W') ;
- une donnée de coureur (masse, horaire de départ) introuvable.

Commence par la phase 0 et montre-moi le résultat avant de passer à la suite.

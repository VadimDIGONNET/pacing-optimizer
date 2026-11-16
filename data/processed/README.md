# Donnees nettoyees (avec sources)

Toutes les valeurs reelles ont une colonne `source_url`. Les hypotheses de
modelisation (CdA, Crr, W', a_lat) ne sont jamais dans ces fichiers : elles
vivent dans le code, avec leur justification.

| Fichier | Contenu | Sources |
|---|---|---|
| `tdf2026_stage16_splits.csv` | Pointages (km 4,8 ; 9,7 ; 18,1 ; arrivee) et heure de depart (CEST) des coureurs. Top 5 a chaque pointage intermediaire, top 20 a l'arrivee. | cyclingstage.com (pointages), bikeraceinfo.com (classement final, 17'14" de Cattaneo au km 9,7), bikeradar.com (heures de depart) |
| `tdf2026_stage16_riders.csv` | Masse et taille publiees. `mass_quality` : `cited` = l'infobox Wikipedia cite une source (equipe, PCS, interview) ; `uncited` = valeur Wikipedia sans reference ; `missing` = pas de valeur. | Wikipedia (API, 2026-11-16), reference citee dans `mass_citation` |
| `tdf2026_stage16_weather_segments.csv` | Densite de l'air et vent par segment de 50 m a l'heure de passage d'Evenepoel. | Open-Meteo archive |

Limites connues :
- Les pointages intermediaires de tous les coureurs (au-dela du top 5) sont
  publies par ProCyclingStats et par le site officiel, dont les conditions
  d'acces excluent les robots : ils n'ont pas ete recuperes automatiquement.
- Deux sources divergent d'une seconde sur l'ecart final de Cattaneo,
  Arensman, Castrillo, Tarling et Martinez (arrondi des centiemes) : la
  valeur de bikeraceinfo (classement complet) est retenue, l'autre est notee.
- Les heures de depart sont celles annoncees la veille (bikeradar) ; la
  table dit "CET" mais la course est en heure d'ete (CEST, UTC+2).

# Phase 7 : CLM candidats pour la validation multi-courses (a valider)

Criteres : CLM individuel recent d'un grand tour, trace GPX publique,
pointages intermediaires publies (au moins le top 5 par pointage sur
cyclingstage, idealement tous les coureurs), meteo disponible sur Open-Meteo
(toujours vrai), profil varie pour tester la physique hors de l'etape 16.

| # | Course | Distance, profil | Pointages | GPX | Interet |
|---|---|---|---|---|---|
| 1 | Tour de France 2025, etape 5, Caen - Caen (9 juillet 2025) | 33 km, plat, vent de la Manche | cyclingstage (top 5 par pointage), bikeraceinfo (classement) | cyclingstage (section GPX) | Plat et vent : teste CdA, densite et vent projete sans l'effet des virages. Evenepoel 36'42", Pogacar a 16" |
| 2 | Giro d'Italia 2025, etape 10, Lucca - Pisa (20 mai 2025) | 28,6 km, plat, pluie | cyclingstage : km 8,3 et 20,5 (top 5) | cyclingstage | Route mouillee : teste la limite de virage (a_lat) en conditions degradees. Hoole 32'30", Tarling a 7" |
| 3 | Tour de France 2024, etape 21, Monaco - Nice (21 juillet 2024) | 33,7 km, La Turbie (8,1 km a 5,6 %) et col d'Eze, descente technique | cyclingstage (top 5 par pointage) | cyclingstage | Le plus proche de l'etape 16 : montee, descente, final plat. Pogacar 45'24" |
| 4 | Tour de France 2024, etape 7, Nuits-Saint-Georges - Gevrey-Chambertin (5 juillet 2024) | 25,3 km, vallonne (cote de Curtil-Vergy) | cyclingstage (top 5 par pointage) | cyclingstage | Vallonne court : teste l'inertie et le lissage sur un profil casse. Evenepoel 28'52" |
| 5 | Vuelta a Espana 2025, etape 18, Valladolid (11 septembre 2025) | 12,2 km (raccourci de 27,2 km) | partiels seulement | cyclingstage | Faible interet : trop court, pointages incomplets. A ecarter sauf besoin d'un cas "sprint" |

Recommandation : 1, 2, 3 et 4. Ils couvrent plat/vent, pluie, montagne et
vallonne, avec les memes sources que l'etape 16 (memes limites : top 5 par
pointage, masses Wikipedia).

Donnees qui manqueront de la meme facon qu'en phase 4 : les pointages de
tous les coureurs (ProCyclingStats et sites officiels interdisent les
robots). Si tu peux exporter ces pages a la main dans data/raw/, le
pipeline les integrera.

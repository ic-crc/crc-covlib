[English](./README.md)

# Exemples en C++

## Exécution des exemples

À partir d'un des répertoires d'exemple, utilisez les commandes suivantes:
```bash
make [CONFIG={debug|release}] # Génération de crc-covlib et du programme d'exemple.

make rebuild [CONFIG={debug|release}] # Regénération de crc-covlib et du programme d'exemple.

make run # Exécution du programme d'exemple une fois généré.

make clean # Destruction des fichiers générés lors de la compilation et des fichiers produits lors de l'exécution du programme d'exemple.
```

## Description des exemples

### antenna-pattern
Exemple démontrant différents paramètres d'antenne dont l'utilisation d'un fichier de patron d'antenne.

### area-results-comparison
Comparaison de résultats entre deux différentes simulations. L'aire de réception d'un troisième objet de type Simulation est utilisée pour la sauvegarde de la différence.

### hello-covlib-dynamic-link
Exemple simple montrant comment instancier la classe Simulation et appeler ses méthodes. Lie crc-covlib de façon dynamique au fichier exécutable.

### hello-covlib-static-link
Autre exemple simple montrant comment instancier la classe Simulation et appeler ses méthodes. Lie crc-covlib de façon statique au fichier exécutable (sous Linux seulement).

### iturp1812-landcover
Exemple utilisant le modèle de propagation UIT-R P.1812 pour la génération d'une couverture radio. Utilise à la fois des données d'élévation de terrain (HRDEM DTM) et de couverture au sol (ESA WorldCover).

### iturp1812-surface
Autre exemple utilisant le modèle de propagation UIT-R P.1812 pour la génération de couvertures, mais utilisant cette fois des données de hauteur de surface (selon la méthode décrite à la section 3.2.2 de l'Annexe 1 d'UIT-R P.1812-7) plutôt que des données de couverture au sol. 

### iturp452v17
Exemple utilisant le modèle de propagation UIT-R P.452-17 pour la génération de couvertures radio. Utilise à la fois des données d'élévation du terrain (CDEM) et de couverture au sol (ESA WorldCover).

### iturp452v18
Exemple utilisant le modèle de propagation UIT-R P.452-18 pour la génération de couvertures radio. Utilise à la fois des données d'élévation de terrain (CDEM) et de couverture au sol (ESA WorldCover).

### secondary-terrain-elev-source
Exemple démontrant l'utilisation d'une source de données secondaire pour l'obtention de l'élévation du terrain lors du calcul de couvertures. Lorsque les données d'élévation en provenance de la source primaire sont manquantes, la source de données secondaire est alors automatiquement utilisée comme mécanisme de secours.

### terrain-elev-cdem
Exemple produisant une couverture haute résolution et une couverture basse résolution à l'aide du modèle de propagation Longley-Rice et de données d'élévation de terrain CDEM («Canadian Digital Elevation Model»).

### terrain-elev-custom
Exemple prduisant une couverture Longley-Rice utilisant le format de données de terrain sur mesure («custom») de crc-covlib.

### terrain-elev-hrdem
Exemple produisant une couverture Longley-Rice utilisant des données d'élévation de terrain haute résolution (1 mètre) de Ressources naturelles Canada.

### terrain-elev-srtm
Exemple produisant une couverture Longley-Rice utilisant des données en format SRTM («Shuttle Radar Topography Mission») pour l'élévation du terrain.

### topography-exports
Vérifie que crc-covlib lit correctement les données d'élévation de terrain, de couverture au sol et d'hauteur de surface en exportant les données lues en format matriciel (fichiers *.bil).


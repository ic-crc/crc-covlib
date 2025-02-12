[English](./README.md)

# Exemples en python

## Installation des paquets requis

Des paquets python de tierces parties sont nécessaires à l'exécution de certains exemples. Ils peuvent être installés en utilisant le fichier _requirements.txt_:
```bash
py -m pip install -r requirements.txt        # sous Windows

python3 -m pip install -r requirements.txt   # sous Linux
```

## Description des exemples

### antenna-pattern
Exemple démontrant différents paramètres d'antenne dont l'utilisation d'un fichier de patron d'antenne.

### area-results-comparison
Comparaison de résultats entre deux différentes simulations. L'aire de réception d'un troisième objet de type Simulation est utilisée pour la sauvegarde de la différence.

### hello-covlib
Exemple simple montrant comment instancier la classe Simulation et appeler ses méthodes.

### iturm2101
Exemple montrant comment générer le patron d'antenne pour une antenne à formation de faisceaux (section 5 de la recommandation UIT-R M.2101-0) en utilisant le sous-paquet _helper_ de crc-covlib.

### iturp1411
Ensemble d'exemples utilisant les modèles de propagation d'UIT-R P.1411 ainsi que des données de réseaux routiers et de bâtiments d'OpenStreetMap.

### iturp1812-landcover
Exemple utilisant le modèle de propagation UIT-R P.1812 pour la génération d'une couverture radio. Utilise à la fois des données d'élévation de terrain (HRDEM DTM) et de couverture au sol (ESA WorldCover).

### iturp1812-surface
Autre exemple utilisant le modèle de propagation UIT-R P.1812 pour la génération de couvertures, mais utilisant cette fois des données de hauteur de surface (selon la méthode décrite à la section 3.2.2 de l'Annexe 1 d'UIT-R P.1812-7) plutôt que des données de couverture au sol. 

### iturp452v17
Exemple utilisant le modèle de propagation UIT-R P.452-17 pour la génération de couvertures radio. Utilise à la fois des données d'élévation du terrain (CDEM) et de couverture au sol (ESA WorldCover).

### iturp452v18
Exemple utilisant le modèle de propagation UIT-R P.452-18 pour la génération de couvertures radio. Utilise à la fois des données d'élévation de terrain (CDEM) et de couverture au sol (ESA WorldCover).

### line-of-sight
Détermine si les antennes d'émission et de réception sont en ligne de visibilité directe ou non. Utilise certaines fonctionnalités du sous-paquet _helper_ de crc-covlib.

### local-ray
Comparaison du temps requis pour effectuer des simulations de couverture séquentiellement et en parallèle en utilisant [Ray](https://www.ray.io/) localement.

### numba-jit-compiling
Certaines fonctionnalités du sous-paquet _helper_ peuvent être compilées en code machine optimizé par l'intermédiaire de  [Numba](https://numba.pydata.org/). La compilation juste-à-temps de Numba est désactivée par défaut dans crc-covlib, cet exemple montre comment l'activer.

### overview
Utilisation des différentes méthodes d'un objet _Simulation_ à titre démonstratif.

### profiles
Exemples démontrant l'export de résultats de simulation et de profils de terrain en fichiers *.csv. Les différents profils sont ensuite affichés à l'écran en utilisant _matplotlib_. Démontre également comment effectuer le calcul de résultats (pertes de propagation, intensité de champ, etc.) en utilisant des données sur mesure («custom data») pour l'élévation du terrain, la couverture au sol et la hauteur de surface.

### secondary-terrain-elev-source
Exemple démontrant l'utilisation d'une source de données secondaire pour l'obtention de l'élévation du terrain lors du calcul de couvertures. Lorsque les données d'élévation en provenance de la source primaire sont manquantes, la source de données secondaire est alors automatiquement utilisée comme mécanisme de secours.

### terrain-elev-sources
Ensemble d'exemples simples produisant des couvertures Longley-Rice et utilisant différentes sources de données d'élévation de terrain:
- Modèle numérique d'élévation du Canada (MNÉC), ou «Canadian Digital Elevation Model» (CDEM) en anglais.
- Format de données de terrain sur mesure («custom data») de crc-covlib.
- Modèle numérique d'élévation de haute résolution (MNEHR), ou «High Resolution Digital Elevation Model» (HRDEM) en anglais.
- Données en format SRTM («Shuttle Radar Topography Mission»).
- Autres données sous forme de fichiers GeoTIFF.

### topography-exports
Vérifie que crc-covlib lit correctement les données d'élévation de terrain, de couverture au sol et d'hauteur de surface en exportant les données lues en format matriciel (fichiers *.bil).

### topography-helper
Utilisation du module _topography_ du sous-paquet _crc_covlib.helper_. Ce module utilise les paquets _rasterio_ et _numpy_ de tierces parties afin d'effectuer la lecture de nombreux formats de fichiers matriciels. Les exemples montrent comment différents fichiers d'élévation du terrain, de couverture au sol et hauteur de surface peuvent être lus et soumis à crc-covlib en tant que données sur mesure («custom data») pour le calcul de couvertures.

[English](./README.md)

# crc-covlib
Bibliothèque logicielle pour la prédiction de couverture et d'interférence radio.

- [Introduction](#introduction)
- [Installation des fichiers de données de l'UIT](#installation-des-fichiers-de-données-de-luit)
- [Utilisation de crc-covlib en python](#utilisation-de-crc-covlib-en-python)
- [Utilisation de crc-covlib en tant que bibliothèque de liens dynamiques Windows ou Linux](#utilisation-de-crc-covlib-en-tant-que-bibliothèque-de-liens-dynamiques-windows-ou-linux)
- [Génération de crc-covlib](#génération-de-crc-covlib)
  - [Sous Windows](#sous-windows)
  - [Sous Linux](#sous-linux)


## Introduction

crc-covlib est une bibliothèque logicielle écrite en C++ pour la prédiction de couverture ou d'interférence radio. Elle inclue différents modèles de propagation et peut être utilisée autant sous Windows que Linux. Une [enveloppe logicielle](./python-wrapper/README_FR.md) est également disponible afin de faciliter son utilisation à partir du langage de programmation python. crc-covlib a été conçue de façon à faciliter l'usage de différents modèles de propagation; une couverture radio pouvant être générée à partir de quelques lignes de code seulement!

crc-covlib offre les fonctionnalités suivantes:
* Sélection entre différents **modèles de propagation**: espace libre, [Longley-Rice](https://its.ntia.gov/research-topics/radio-propagation-software/itm/itm.aspx), [UIT-R P.1812-7](https://www.itu.int/rec/R-REC-P.1812/fr), [UIT-R P.452-17](https://www.itu.int/rec/R-REC-P.452/fr), [UIT-R P.452-18](https://www.itu.int/rec/R-REC-P.452/fr) et [eHata](https://its.ntia.gov/about-its/archive/2017/its-open-sources-the-extended-hata-urban-propagation-model.aspx). D'autres modèles en lien à la propagation sont aussi disponibles tels que le modèle statistique de l'affaiblissement dû à un groupe d'obstacles pour des trajets terrestres (section 3.2 de l'Annexe 1 d'[UIT-R P.2108-1](https://www.itu.int/rec/R-REC-P.2108/fr)), le modèle de prévision de l'affaiblissement dû à la pénétration dans les bâtiments (Annexe 1 d'[UIT-R P.2109-2](https://www.itu.int/rec/R-REC-P.2109/fr)) ainsi que le modèle d'affaiblissement dû aux gaz de l'atmosphère (sections 1 et 2.1 de l'Annexe 1 d'[UIT-R P.676-13](https://www.itu.int/rec/R-REC-P.676-13-202208-I/fr)).
* Lecture de fichiers de données d'**élévation du terrain** selon la résolution d'échantillonnage définie par l'usager. Les fichiers de données étant supportés incluent le Modèle numérique d'élévation du Canada (MNEC), le Modèle numérique d'élévation de moyenne résolution (MNEMR MNT) et le Modèle numérique d'élévation haute résolution (MNEHR MNT) de Ressources naturelles Canada. D'autres fichiers en format GeoTIFF peuvent également être utilisés dépendamment du système de coordonnées (WGS84 est supporté pour fins de simulations à n'importe quel emplacement terrestre).
* Lecture de fichiers de données de **couverture au sol** (obstacles) du produit WorldCover de l'Agence Spatiale Européenne (ASE) et du produit Couverture des terres du Canada de Ressources naturelles Canada. D'autres fichiers en format GeoTIFF peuvent également être utilisés dépendamment du système de coordonnées (WGS84 est supporté pour fins de simulations à n'importe quel emplacement terrestre).
* Lecture de fichiers de données d'**hauteur de surface**, incluant le Modèle numérique de surface du Canada (MNSC), le Modèle numérique d'élévation de moyenne résolution (MNEMR MNS) et le Modèle numérique d'élévation haute résolution (MNEHR MNS) de Ressources naturelles Canada. D'autres fichiers en format GeoTIFF peuvent être utilisés dépendamment du système de coordonnées (WGS84 est supporté pour fins de simulations à n'importe quel emplacement sur terre). Le format de fichier SRTM («Shuttle Radar Topography Mission») est aussi supporté.
* Spécification de sources de données secondaires pour suppléer aux sources de données principales (élévation du terrain, couverture au sol et hauteur de surface) aux emplacements où les données peuvent être absentes.
* Utilisation de patrons d'antenne possible aux deux terminaux (émetteur et récepteur). L'angle d'élévation pour l'obtention du gain au patron vertical est calculé selon les données d'élévation du terrain. L'antenne peut être pointée en azimut fixe ou relativement à l'autre terminal.
* Résultats de simulations en intensité de champ (dBµV/m), perte de propagation (dB), pertes de transmission (dB) ou puissance reçue (dBm).
* Résultats de simulations point-à-point générés en format texte (\*.txt) et matriciel (\*.bil). Zones de coutour générées en format *.kml (Google Earth) et *.mif/mid (MapInfo).
* Génération de profils de terrain (élévation du terrain, couverture au sol, etc.) pouvant être sauvegardés en format *.csv.
* D'autres fonctionnalités disponibles seulement pour le langage de programmation python sont aussi offertes en plus de celles implémentées en C++, dont des implémentations addtionnelles de recommendations de l'UIT (propagation terrestre et non-terrestre). Pour de plus amples détails, veuillez consulter la [documentation du paquet crc_covlib](./python-wrapper/README_FR.md#documentation-du-paquet-crc_covlib).


<p align = "center">
<img src="coverage_example.png" alt="drawing" width="700"/>
</p>
<p align = "center">
Figure 1 - Zones de contour produites par crc-covlib, affichées à l'aide du logiciel <a href="https://qgis.org/fr/site/">QGIS</a> avec image de fond d'OpenStreetMap
</p>


## Installation des fichiers de données de l'UIT

crc-covlib utilise des cartes digitales et d'autres fichiers de données publiquement accessibles sur le site web de l'[Union internationale des télécommunications](https://www.itu.int/fr/Pages/default.aspx#/fr) (UIT). Ces fichiers ne sont pas directement redistribuables, mais ils peuvent tout de même être facilement obtenus en exécutant le script python `install_ITU_data.py` qui se chargera du téléchargement en provenance de la source officielle. L'option d'installation par défaut du script télécharge à la fois les fichiers utilisés par l'implémentation C++ et ceux utilisés par les fonctionalités additionelles de l'enveloppe logicielle python. Si vous prévoyez n'utiliser que les fonctionalités principales listés ci-haut, vous pouvez alors sélectionner l'option d'installation minimale.

Les fichiers de données de l'UIT sont pour usage personnel seulement. Ils peuvent être installés à partir de l'invite de commande:
```bash
py install_ITU_data.py       # sous Windows
python3 install_ITU_data.py  # sous Linux
```


## Utilisation de crc-covlib en python

Pour les détails concernant l'utilisation de crc-covlib en python, veuillez vous référer à [cette page](./python-wrapper/README_FR.md).

Des exemples de code en python sont disponibles sous le répertoire [python-wrapper/examples](./python-wrapper/examples/).


## Utilisation de crc-covlib en tant que bibliothèque de liens dynamiques Windows ou Linux

Le répertoire de distribution [dist](./dist/) contient la plus récente version compilée de crc-covlib pour l'environnement Windows. Ce répertoire comporte les fichiers suivants:

* CRC-COVLIB.h (fichier d'en-tête à inclure dans tout projet C++ utilisant crc-covlib)
* crc-covlib.dll (bibliothèque de liens dynamiques Windows)
* crc-covlib.lib (bibliothèque d'importation)

Pour l'environnement Linux, une bibliothèque de liens dynamiques (libcrc-covlib.so) et une bibliothèque de liens statiques (libcrc-covlib.a) peuvent être générées en suivant les [instructions ci-bas](#sous-linux).

Des exemples de code C++ utilisant crc-covlib sont disponibles sous le répertoire [examples](./examples/).

Pour plus de détails, veuillez vous référer à la documentation sur l'[interface de programmation (API)](./docs/CRC-COVLIB%20API%20Reference.pdf) (anglais seulement).


## Génération de crc-covlib

### Sous Windows

Sous Windows, le fichier `crc-covlib.dll` est présentement généré à l'aide de [MinGW](https://www.mingw-w64.org/). MinGW peut être facilement installé via [MSYS2](https://www.msys2.org/). Après avoir complété la procédure d'installation fournie sur le site web de MSYS2, installez la suite d'outils suivante à partir de l'invite de commande MSYS2 UCRT64:
```
pacman -S mingw-w64-ucrt-x86_64-toolchain
```

Installez ensuite les fichiers de développement pour [LibTIFF](http://libtiff.maptools.org/) et [GeographicLib](https://geographiclib.sourceforge.io/) sur la même invite de commande:
```
pacman -S mingw-w64-ucrt-x86_64-libtiff
pacman -S mingw-w64-ucrt-x86_64-geographiclib
```

Ouvrez une invite de commande Windows en mode Administrateur et créez un lien symbolique pour la commande `make` (_note: modifiez les répertoires ci-dessous si les répertoires définis par défaut lors de l'installation de MSYS2 ont été modifiés_):
```
mklink C:\msys64\ucrt64\bin\make.exe C:\msys64\ucrt64\bin\mingw32-make.exe
```

Optionnellement, ajoutez les répertoires suivants (_note: encore une fois, ajustez les répertoires si des changements aux répertoires par défaut ont été faits pendant l'installation de MSYS2_) à la variable d'environnement `PATH` afin de rendre `g++` et `make` disponibles sous l'invite de commande Windows. Sinon, vous pourrez toujours invoquer `g++` et `make` à partir de l'invite de commande MSYS2 UCRT64.
```
C:\msys64\ucrt64\bin
C:\msys64\usr\bin
```

Finalement, utilisez la commande `make` à partir du répertoire contenant le fichier [Makefile](./Makefile) pour générer le fichier de bibliothèque de liens dynamiques (DLL):

```
make rebuild
```

Note: assurez-vous de [mettre à jour votre installation de MSYS2](https://www.msys2.org/docs/updating/) de temps à autre en invoquant la commande  `pacman -Suy` à **deux** reprises à partir de l'invite de commande MSYS2 MSYS.


### Sous Linux

Sous Linux, la suite d'outils GNU peut être utilisée pour compiler et générer crc-covlib. Tout d'abord, installez les fichiers de développement pour [LibTIFF](http://libtiff.maptools.org/) et [GeographicLib](https://geographiclib.sourceforge.io/) (les commandes peuvent différer selon la distribution Linux et le gestionnaire de paquets installé):
```bash
# Ubuntu et dérivés
sudo apt install libtiff-dev
sudo apt install libgeographiclib-dev # note: si libgeographiclib-dev n'est pas disponible, essayez plutôt libgeographic-dev

# CentOS, Amazon Linux 2
sudo yum install libtiff-devel
sudo yum install GeographicLib-devel
```

Enfin, invoquez la commande `make` à partir du répertoire contenant le fichier [Makefile](./Makefile) afin de générer les fichiers `libcrc-covlib.so` et `libcrc-covlib.a`.
```
make rebuild
```

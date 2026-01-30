[English](./README.md)

# Tests C++

## Exécution des programmes de test

À partir d'un des répertoires de test, utilisez les commandes suivantes:
```bash
make [CONFIG={debug|release}] # Génération de crc-covlib (si nécessaire) et du programme de test.

make rebuild [CONFIG={debug|release}] # Regénération de crc-covlib (si nécessaire) et du programme de test.

make run # Exécution du programme de test une fois généré.

make clean # Destruction des fichiers générés lors de la compilation et des fichiers produits lors de l'exécution du programme de test.
```
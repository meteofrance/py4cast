# PNIA : Prévision numérique par IA

Projet administré par DSM/Lab IA et CNRM/GMAP/PREV

MaJ : 11/12/2023

## Utilisation

Pour l'instant ce projet doit être utilisé avec l'outil `runai` à l'intérieur du monorépo du Lab IA.

Pour ce faire, clonez le monorépo du Lab : [lien](https://git.meteo.fr/dsm-labia/monorepo4ai)

`pnia` est un submodule du monorépo, accessible dans le dossier `projects/pnia`.

Les commandes `runai` doivent être lancées à la racine du dossier `pnia` :

```runai build```  -> build de l'image Docker

```runai python pnia/titan_dataset.py``` -> lancement d'un script python

```runai python_mpl pnia/plots_grid.py``` -> lancement d'un script python avec plot

## Entrainement

```bash
runai gpu_play 4

runai exec_gpu python bin/prepare.py smeagol grid # Préparation des statics smeagol

runai exec_gpu python bin/prepare.py nlam --dataset smeagol # Construction des pré-requis pour les graphes

# Avec un réseau de type Graphe
runai exec_gpu python bin/train.py --dataset smeagol --model graph --standardize --gpus 4

# Avec un réseau convolutif (par défaut HalfUnet)
runai exec_gpu python bin/train.py --dataset smeagol --model conv --standardize --gpus 4
```
[More information here](./bin/Readme.md)

## Architecture

- Le dossier `submodules` contient des submodules (au sens git) de plusieurs répo open source de codes de PN par IA (Pangu, ClimaX, Neural-LAM,...). On peut ainsi facilement importer des fonctions issues de ces projets dans nos codes.

- Le dossier `pnia` contient pour le moment les codes servant à faire fonctionner neural-LAM avec le jeu de données Titan.

## Prochaines étapes

- [ ]  merger train_vincent et train
- [ ]  pourquoi les nan sur Titan ?
- [ ]  tous les TODO du Titan dataset (flux, stadardisation)
- [ ]  meilleure séparation model / trainer
- [ ]  get item fonctionnel pour tout type de modèle (pb grille graph)
- [ ]  faire marcher avec un U-Net
- [ ]  faire marcher AIFS
- [ ]  meilleur logger
- [ ]  comment gérer la CLI
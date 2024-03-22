# PNIA : Prévision numérique par IA

Ce projet fourni un cadre **PyTorch** et **PyTorch-lightning** pour entrainer des architectures variées (Graph NN, Réseaux convolutifs, Transformers, ...) sur des jeux de données.

Projet administré par DSM/Lab IA et CNRM/GMAP/PREV

## Structure du projet

```
bin => scripts principaux
config => Fichiers de configuration
pnia => package python du projet
Dockerfile => recette docker de construction du contexte d'éxecution
requirements.txt => dépendances python
.gitlab-ci.yml => intégration continue gitlab
```

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
```

Currently we support the following neural network architectures: hilam, halfunet and segformer. 
To train on a dataset using a network with its default settings just pass the name of the architecture
(all lowercase) as shown below:

```bash
runai exec_gpu python bin/train.py --dataset smeagol --model hilam --gpus 4

runai exec_gpu python bin/train.py --dataset smeagol --model halfunet --gpus 4

runai exec_gpu python bin/train.py --dataset smeagol --model segformer --gpus 4
```

You can override some settings of the model using a json config file (here we increase the number of filter to 128 and use ghost modules):

```bash
runai exec_gpu python bin/train.py --dataset smeagol --model halfunet --gpus 1 --model_conf config/halfunet128_ghost.json
```

You can also override the dataset default configuration file:

```bash
runai exec_gpu python bin/train.py --dataset smeagol --model halfunet --gpus 4 --data_conf config/smeagol.json
```

[More information here](./bin/Readme.md)

## Architecture

- Le dossier `submodules` contient des submodules (au sens git) de plusieurs répo open source de codes de PN par IA (Pangu, ClimaX, Neural-LAM,...). On peut ainsi facilement importer des fonctions issues de ces projets dans nos codes.

- Le dossier `pnia` contient pour le moment les codes servant à faire fonctionner neural-LAM avec le jeu de données Titan.

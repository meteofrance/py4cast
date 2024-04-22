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
runai exec_gpu python bin/train.py --dataset smeagol --model hilam

runai exec_gpu python bin/train.py --dataset smeagol --model halfunet

runai exec_gpu python bin/train.py --dataset smeagol --model segformer
```

You can override some settings of the model using a json config file (here we increase the number of filter to 128 and use ghost modules):

```bash
runai exec_gpu python bin/train.py --dataset smeagol --model halfunet --model_conf config/halfunet128_ghost.json
```

You can also override the dataset default configuration file:

```bash
runai exec_gpu python bin/train.py --dataset smeagol --model halfunet --dataset_conf config/smeagol.json
```

[More information here](./bin/Readme.md)

## Available PyTorch's architecture

| Model  | Research Paper  | Input Shape    | Notes  | Maintainer(s) |
| :---:   | :---: | :---: | :---: | :---: |
| halfunet | https://www.researchgate.net/publication/361186968_Half-UNet_A_Simplified_U-Net_Architecture_for_Medical_Image_Segmentation | (Batch, Height, Width, features)   | In prod/oper on espresso v2 with 128 filters and standard conv blocks instead of ghost |  Frank Guibert |
| segformer | https://arxiv.org/abs/2105.15203   | (Batch, Height, Width, features) | on par with u-net like on deepsyg, added an upsampling stage. Adapted from [Lucidrains' github](https://github.com/lucidrains/segformer-pytorch) |  Frank Guibert |
| hilam, graphlam | https://arxiv.org/abs/2309.17370  | (Batch, graph_node_id, features)   | Imported and adapted from [Joel's github](https://github.com/joeloskarsson/neural-lam) |  Vincent Chabot/Frank Guibert |

## Architecture

- Le dossier `submodules` contient des submodules (au sens git) de plusieurs répo open source de codes de PN par IA (Pangu, ClimaX, Neural-LAM,...). On peut ainsi facilement importer des fonctions issues de ces projets dans nos codes.

- Le dossier `pnia` contient pour le moment les codes servant à faire fonctionner neural-LAM avec le jeu de données Titan.

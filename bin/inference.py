from lightning.pytorch.cli import LightningCLI, ArgsType
from py4cast.ARLightningModule import AutoRegressiveLightningModule
from py4cast.TitanDataModule import TitanDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import os

def find_available_checkpoint_name(dirpath, model_name, index=0):
    # Créer le nom du fichier
    checkpoint_name = f"best_{model_name}_{index}.ckpt"
    checkpoint_path = os.path.join(dirpath, checkpoint_name)

    # Vérifier si le fichier existe
    if not os.path.exists(checkpoint_path):
        return checkpoint_name
    else:
        # Si le fichier existe, appeler la fonction récursivement avec l'index incrémenté
        return find_available_checkpoint_name(dirpath, model_name, index + 1)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.dataset_name", "model.dataset_name")
        parser.link_arguments(
            "data.train_dataset_info", "model.dataset_info", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.batch_shape", "model.batch_shape", apply_on="instantiate"
        )
        parser.add_argument("--checkpoint_path", type=str, default="/scratch/shared/py4cast/models/best.ckpt", help="Path to the best checkpoint")


def cli_main(args: ArgsType = None):
    cli = MyLightningCLI(
        AutoRegressiveLightningModule,
        TitanDataModule,
        run=False,
        args=args,
    )
    model_name = cli.model.__class__.__name__.lower()
    checkpoint_dir = args.checkpoint_path.rstrip('/')  # Répertoire où sauvegarder les checkpoints
    checkpoint_filename = find_available_checkpoint_name(checkpoint_dir, model_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Changez cela selon la métrique que vous souhaitez surveiller
        save_top_k=1,  # Sauvegarder le meilleur checkpoint
        mode='min',  # 'min' pour val_loss, 'max' pour val_accuracy
        dirpath=checkpoint_dir,  # Répertoire où sauvegarder les checkpoints
        filename=checkpoint_filename  # Nom du fichier du meilleur modèle
    )
    cli.trainer.callbacks.append(checkpoint_callback)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    cli.trainer.test(cli.model, cli.datamodule.test_dataloader(), ckpt_path=checkpoint_path) # Lance l'inférence


if __name__ == "__main__":
    cli_main(
        [
            "--config=config/config_cli_inference.yaml",
            "--checkpoint_path=/scratch/shared/py4cast/models/",
        ]
    )

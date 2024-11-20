"""
Main script to use the model with lightning CLI
Training with fit and infer with predict
"""

from py4cast.cli import cli_main

# python bin/launcher.py fit --config config/CLI/trainer.yaml --config config/CLI/poesy.yaml --config config/CLI/halfunet.yaml

# python -m pdb bin/launcher.py predict --ckpt_path /scratch/shared/py4cast/logs/test_cli/ --config config/CLI/trainer.yaml --config config/CLI/poesy.yaml --config config/CLI/halfunet.yaml

if __name__ == "__main__":
    cli_main()

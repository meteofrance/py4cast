"""
Main script to use the model with lightning CLI
Training with fit and infer with predict
Exemple usage:
    runai python bin/main.py fit --config bin/config_test_cli.yaml
"""

from py4cast.cli import Py4castLightningCLI
from py4cast.lightning import AutoRegressiveLightning, PlDataModule

if __name__ == "__main__":
    Py4castLightningCLI(AutoRegressiveLightning, PlDataModule)

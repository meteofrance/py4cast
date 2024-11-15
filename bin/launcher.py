from py4cast.cli import cli_main

# Train : python bin/launcher.py fit --config config/CLI/exp_alpha.yaml

# Inference : python -m pdb bin/launcher.py predict --config config/CLI/exp_alpha.yaml --ckpt_path /scratch/shared/py4cast/logs/test_cli/last.ckpt
if __name__ == "__main__":
    cli_main()

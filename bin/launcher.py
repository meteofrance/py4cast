from py4cast.cli import cli_main

# Launch : python bin/train_cli.py fit --config config/CLI/test.yaml

# Inference : python -m pdb bin/launcher.py predict --config config/CLI/test.yaml --ckpt_path /scratch/shared/py4cast/logs/test_cli/last.ckpt
if __name__ == "__main__":
    cli_main()

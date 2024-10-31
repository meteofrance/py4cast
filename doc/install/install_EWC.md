## Installation instruction on EWC ECMWF A100 machines

This procedure uses MF's docker wrapper runai and assumes it is available on your PATH and that docker has been installed on your machine.

```bash
export RUNAI_DOCKER_FILENAME=Dockerfile.ewc_flash_attn
runai build
```

You should now be able to run a test training with the Dummy dataset using flash_attn and bf16 precision:

```bash
 runai exec_gpu python bin/train.py --dataset dummy --model unetrpp --epochs 1  --batch_size 2 --model_conf config/models/unetrpp161024_linear_up_flash.json --precision bf16
 ```
 


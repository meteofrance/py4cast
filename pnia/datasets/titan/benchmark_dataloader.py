"""
Utility script to measure the performances
of our dataloader when using multiple CPU cores
"""
from itertools import cycle
from torch.utils.data import DataLoader
from mfai.torch.transforms import ToTensor
from torchvision import transforms
from pnia.datasets.titan import TitanDataset, TitanHyperParams
import time

if __name__ == "__main__":
    
    from argparse_dataclass import ArgumentParser

    parser = ArgumentParser(TitanHyperParams)
    hparams = parser.parse_args()
    batch_size=4

    print("hparams : ", hparams)
    dataset = TitanDataset(hparams)

    loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=hparams.num_workers,
            shuffle=True,
            # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/2
            drop_last=True,
            pin_memory=True,
        )

    start_time = time.time()
    no_iters = 100
    data_iter = iter(loader) 
    for i in range(no_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        print(i)

    print(i)
    stop_time = time.time()
    delta = stop_time - start_time
    speed = no_iters*batch_size/delta
    print(f"Loading speed: {speed} sample(s)/sec")
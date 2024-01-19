"""
Utility script to measure the performances
of our dataloader when using multiple CPU cores
"""
from torch.utils.data import DataLoader
from mfai.torch.transforms import ToTensor
from torchvision import transforms
from pnia.datasets.titan import TitanDataset, TitanHyperParams
import time

if __name__ == "__main__":
    
    from argparse_dataclass import ArgumentParser

    parser = ArgumentParser(TitanHyperParams)
    hparams = parser.parse_args()

    print("hparams : ", hparams)
    dataset = TitanDataset(hparams)
    transform = transforms.Compose(
            [
                ToTensor(),
            ]
        )
    loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=hparams.num_workers,
            shuffle=False,
            # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/2
            drop_last=True,
        )

    count = 0
    start_time = time.time()
    nb_samples = 1000
    for _ in loader:
        count += 1
        if count >= nb_samples:
            break
    stop_time = time.time()
    delta = stop_time - start_time
    speed = nb_samples/delta
    print(f"Loading speed: {speed} sample(s)/sec")
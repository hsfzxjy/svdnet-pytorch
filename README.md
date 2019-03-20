# SVDNet-pytorch

Based on ResNet-50 baseline from [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid). Original README at [here](./README_orig.md).

Paper at [here](https://arxiv.org/pdf/1703.05693.pdf).

## Training

Please refer to [Original README](./README_orig.md) for data preparation.

```bash
python3 train_svdnet_xent.py --save-dir path/to/dir --gpu-devices 0
```


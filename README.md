# simple-grad
Pytorch-style autograd library written in numpy.

## Setup 
```bash 
# inside a python virtualenv
pip install -e .
```

## Development
Run unittest:
```
pytest tests/
```

## MNIST example

Download and save MNIST locally once:

```bash
python scripts/download_mnist.py
```

The script skips the download when a valid dataset already exists at
`tmp/data/mnist`. Then run training with:

```bash
python examples/train_mnist.py
```

Select a Conv2d backend with `--conv-implementation patch`, `strided`, or
`im2col`:

```bash
python examples/train_mnist.py --conv-implementation strided
```

Compare the patch-loop, strided-window, and full-im2col Conv2d backends:

```bash
python scripts/benchmark_conv2d.py
```

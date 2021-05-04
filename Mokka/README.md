# MNIST test

```./DOWNLOAD_MNIST.sh``` if you don't have it (linux/WSL only). Or get MNIST manually, files must be named as:

```
./mnist/t10k-images.idx3-ubyte
./mnist/t10k-labels.idx1-ubyte
./mnist/train-images.idx3-ubyte
./mnist/train-labels.idx1-ubyte
```

Then.
```bash
dotnet run -c release
```

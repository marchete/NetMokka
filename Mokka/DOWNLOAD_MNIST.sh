#!/bin/sh
echo "MNIST Download"
FILE=./mnist/train-images.idx3-ubyte 
if [ -f "$FILE" ]; then
    echo "$FILE already exists. Cancelling."
	exit
fi
mkdir mnist
cd mnist
wget https://data.deepai.org/mnist.zip
unzip mnist.zip
gunzip *.gz


mv t10k-images-idx3-ubyte t10k-images.idx3-ubyte
mv t10k-labels-idx1-ubyte t10k-labels.idx1-ubyte
mv train-images-idx3-ubyte train-images.idx3-ubyte
mv train-labels-idx1-ubyte train-labels.idx1-ubyte

if [ -f "$FILE" ]; then
	rm *.zip
fi
cd ..
echo "Done"

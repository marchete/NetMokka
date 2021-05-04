using System;
using System.IO;
using System.Linq;

namespace mnist {
//https://github.com/Rosti81/MNIST/blob/master/MNIST

 	public class Digit
    {
        public int Number { get; set; }
        public float[] Pixels { get; set; }
    }	

public static class MNIST_COMMON{
        static private readonly string _folderFiles = "mnist";

        static private readonly string _trainingSetLabelsFile = "train-labels.idx1-ubyte";
        static private readonly string _trainingSetImagesFile = "train-images.idx3-ubyte";
        static public Digit[] _trainingDigits = new Digit[60000];

        static private readonly string _testSetLabelsFile = "t10k-labels.idx1-ubyte";
        static private readonly string _testSetImagesFile = "t10k-images.idx3-ubyte";
        static public Digit[] _testDigits = new Digit[10000];

        public static void LoadTrainingDigits()
        {
            _trainingDigits = new Digit[60000];

            var fileBytes = File.ReadAllBytes(_folderFiles+Path.DirectorySeparatorChar+_trainingSetLabelsFile);
            var i = 0;
            foreach (var item in fileBytes.Skip(8))
            {
                _trainingDigits[i] = new Digit
                {
                    Number = (int)item,
                    Pixels = new float[784]
                };
                i++;
            }

            fileBytes = File.ReadAllBytes(_folderFiles+Path.DirectorySeparatorChar+_trainingSetImagesFile);
            var j = i = 0;
            foreach (var item in fileBytes.Skip(16))
            {
                _trainingDigits[i].Pixels[j] = item;
                j++;
                if (j == 784)
                {
                    i++;
                    j = 0;
                }
            }
        }

        public static void LoadTestDigits()
        {
            var fileBytes = File.ReadAllBytes(_folderFiles+Path.DirectorySeparatorChar+_testSetLabelsFile);
            var i = 0;
            foreach (var item in fileBytes.Skip(8))
            {
                _testDigits[i] = new Digit
                {
                    Number = (int)item,
                    Pixels = new float[784]
                };
                i++;
            }


            fileBytes = File.ReadAllBytes(_folderFiles+Path.DirectorySeparatorChar+_testSetImagesFile);
            var j = i = 0;
            foreach (var item in fileBytes.Skip(16))
            {
                _testDigits[i].Pixels[j] = item;
                j++;
                if (j == 784)
                {
                    i++;
                    j = 0;
                }
            }
        }
}
	
}
using System;
using System.Collections.Generic;
using System.Diagnostics;
using NetMokka;
using mnist;

using System.Runtime.Intrinsics;

namespace Prog
{

    class Program
    {

        static Model CreateModel(out Input input,out Layer policy, out Layer value ){
            Layer x;
            input = new Input(new List<int>(){28*28});
            x     = (new Dense("Dense",128,Activators.RELU)).link(input);
            policy = (new Dense("Soft", 10,Activators.SOFTMAX)).link(x);
	        Model model = new Model( input,  policy );
            value = null;
	        return model;
        }
        static Model CreateModel29(out Input input,out Layer policy, out Layer value ){
            Layer x;
            input = new Input(new List<int>(){29*29});
            x     = (new Dense("Dense",127,Activators.RELU)).link(input);
            policy = (new Dense("Soft", 13,Activators.SOFTMAX)).link(x);
	        Model model = new Model( input,  policy );
            value = null;
	        return model;
        }


	static Tensor imageToTensorFn( Digit d,int padding =0)
	{
		Tensor t = new Tensor( new List<int>(){ 1,1, (28+padding) * (28 + padding) });
		for (int y = 0; y < 28; ++y) {
			for (int x = 0; x < 28; ++x) {
				t.setElement(y *(28 + padding) + x, d.Pixels[y * 28 + x] / 255.0f);
			}
		} 
        return t;
	}

	static float calcAccuracy(Model model,List<Tensor> t_images,Digit[] t_digits)
	{
		int total = 0;
		int correct = 0;
		for (int index = 0; index < t_images.Count; ++index)
		{
			model.predict( t_images[index]);
			var ans = model.outputs[0].output;
			float MaxValue = -9999999.99f;
			int maxIndex = -1;
			for (int i = 0; i < ans.size; ++i)
			{
				if (ans.getElement(i) > MaxValue) {
					MaxValue = ans.getElement(i);
					maxIndex = i;
				}
			}
			if (t_digits[index].Number == maxIndex)
			{
				correct++;
			}
			total++;
		}
		return  ((float)correct)*100.0f / (float)total;
	}

        static void MNIST_inference(Model model,int padding =0){
            List<Tensor> Tensor_training_images = new List<Tensor>();
            List<Tensor> Tensor_test_images= new List<Tensor>();
            Console.Error.WriteLine( "Nbr of training images = " + MNIST_COMMON._trainingDigits.Length);
            Console.Error.WriteLine( "Nbr of test images = " + MNIST_COMMON._testDigits.Length );
            foreach (var d in MNIST_COMMON._trainingDigits)
                {
                    Tensor_training_images.Add(imageToTensorFn(d,padding));
                }
            foreach (var d in MNIST_COMMON._testDigits)
                {
                    Tensor_test_images.Add(imageToTensorFn(d,padding));
                }
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Restart();
            double f = 0.0;
            double testAcc = 0.0f;
            double trainAcc = 0.0f;
            for (int i = 0; i < 10; ++i)
            {
                testAcc = calcAccuracy(model,Tensor_test_images,MNIST_COMMON._testDigits) ;
                trainAcc =  calcAccuracy(model,Tensor_training_images,MNIST_COMMON._trainingDigits) ;                
                f+= testAcc +trainAcc;
            }         
            int countPredict  = 10*(Tensor_training_images.Count+Tensor_test_images.Count);
            Console.Error.WriteLine(" Took: " + stopwatch.ElapsedMilliseconds + "ms mean:" + stopwatch.ElapsedMilliseconds*1000 / countPredict + "us/sample");
            Console.Error.WriteLine( "Est Acc:" + testAcc.ToString("#.##") + "% TrainingAcc:" + trainAcc.ToString("#.##") + "%");
        }

/*https://github.com/CBGonzalez/Core3Intrinsics-Intro
System.Runtime.CompilerServices.Unsafe.CopyBlock(ref byte destination, ref byte source, uint byteCount)
        public float[] ProcessData(ref Span<float> input)
        {
            float[] results = new float[input.Length];
            Span<Vector256<float>> resultVectors = MemoryMarshal.Cast<float, Vector256<float>>(results);

            ReadOnlySpan<Vector256<float>> inputVectors = MemoryMarshal.Cast<float, Vector256<float>>(input);

            for(int i = 0; i < inputVectors.Length; i++)
            {
                resultVectors[i] = Avx.Sqrt(inputVectors[i]);                
            }

            return results;
        }*/

        unsafe static void Main(string[] args)
        {
            var T1 = Vector256.Create(1.0f,10.0f,100.0f,1000.0f,10000.0f,100000.0f,100500.0f,105000.0f);
            NN_utils.hsums(T1);
            Input input;
	        Layer policy,value;
            MNIST_COMMON.LoadTestDigits();
            MNIST_COMMON.LoadTrainingDigits();

            Console.Error.WriteLine( "*****************************************************");
            Console.Error.WriteLine( "********** TEST 'MNIST Simple.ipynb' ****************");
            Console.Error.WriteLine( "********** Using dense.weights       ****************");
            Console.Error.WriteLine( "*****************************************************");

            Model model = CreateModel(out input,out policy,out value);
            model.summary();
            model.loadWeights("DENSE.weights");
            model.saveWeights("DENSE.test");
            MNIST_inference(model);

            Console.Error.WriteLine( "*****************************************************");
            Console.Error.WriteLine( "********** TEST 'MNIST Simple29.ipynb' ****************");
            Console.Error.WriteLine( "********** Using dense29.weights       ****************");
            Console.Error.WriteLine( "*****************************************************");

            Model model29 = CreateModel29(out input,out  policy,out  value);
            model29.summary();
            model29.loadWeights("DENSE29.weights");
            model29.saveWeights("DENSE29.test");
            MNIST_inference(model29,1);


        }
    }
}

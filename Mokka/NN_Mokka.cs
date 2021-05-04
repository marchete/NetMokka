using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.CompilerServices;

using __m256 = System.Runtime.Intrinsics.Vector256<float>;

namespace NetMokka{

static class NN_utils{

[MethodImpl(MethodImplOptions.AggressiveInlining)]
static __m256 SET(float f){return Vector256.Create(f);}

#region fast exponentials	
	static readonly __m256 EXP_C1 = SET(1064872507.1541044f);
	static readonly __m256 EXP_C2 = SET(12102203.161561485f);
 [MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static __m256 fast_exp256_ps(__m256 V) {
		return Vector256.AsSingle(Avx2.ConvertToVector256Int32WithTruncation(Fma.MultiplyAdd(EXP_C2,V,EXP_C1)));
	}

static readonly __m256 exp_hi = SET(88.3762626647949f);
static readonly __m256 exp_lo = SET(-88.3762626647949f);
static readonly __m256 cLOG2EF = SET(1.44269504088896341f);
static readonly __m256 cexp_C1 = SET(0.693359375f);
static readonly __m256 cexp_C2 = SET(-2.12194440e-4f);
static readonly __m256 cexp_p0 = SET(1.9875691500E-4f);
static readonly __m256 cexp_p1 = SET(1.3981999507E-3f);
static readonly __m256 cexp_p2 = SET(8.3334519073E-3f);
static readonly __m256 cexp_p3 = SET(4.1665795894E-2f);
static readonly __m256 cexp_p4 = SET(1.6666665459E-1f);
static readonly __m256 cexp_p5 = SET(5.0000001201E-1f);
 //[MethodImpl(MethodImplOptions.AggressiveInlining)]
 public static __m256 exp256_ps( __m256 V) {
	__m256 x = V;
	__m256 tmp =  __m256.Zero;
	__m256 one = SET(1.0f);
	x = Avx2.Min(x, exp_hi);
	x = Avx2.Max(x, exp_lo);
	__m256 fx = Avx2.Multiply(x, cLOG2EF);
	fx = Avx2.Add(fx, SET(0.5f));
	tmp = Avx2.Floor(fx);
	var mask = Avx2.Compare(tmp, fx,FloatComparisonMode.OrderedGreaterThanSignaling);
	mask = Avx2.And(mask, one);
	fx = Avx2.Subtract(tmp, mask);
	tmp = Avx2.Multiply(fx, cexp_C1);
	__m256 z = Avx2.Multiply(fx, cexp_C2);
	x = Avx2.Subtract(x, tmp);
	x = Avx2.Subtract(x, z);
	z = Avx2.Multiply(x, x);
	__m256  y = cexp_p0;
	y = Fma.MultiplyAdd(y, x, cexp_p1);
	y = Fma.MultiplyAdd(y, x, cexp_p2);
	y = Fma.MultiplyAdd(y, x, cexp_p3);
	y = Fma.MultiplyAdd(y, x, cexp_p4);
	y = Fma.MultiplyAdd(y, x, cexp_p5);
	y = Fma.MultiplyAdd(y, z, x);
	y = Avx2.Add(y, one);
	var imm0  = Avx2.ConvertToVector256Int32(fx);
	var F7 = Vector256.Create((int)0x7f);
	imm0 = Avx2.Add(imm0,F7);
	imm0 = Avx2.ShiftLeftLogical(imm0, 23);
	__m256 pow2n = Vector256.AsSingle(imm0);
	y = Avx2.Multiply(y, pow2n);
	return y;
}

#endregion

// Does horizontal sum of a chunk v
[MethodImpl(MethodImplOptions.AggressiveInlining)] 
public static __m256 hsums(__m256 v) {
	var ymm2 = Avx2.Permute2x128(v,v,1);
	var ymm = Avx2.Add(v,ymm2);
	ymm = Avx2.HorizontalAdd(ymm,ymm);
	ymm = Avx2.HorizontalAdd(ymm,ymm);
    return ymm;  	
}

}


unsafe class Tensor {
	public __m256[] xmm = null;
	public float* pFloats;
	public int size= 0;
	public List<int> shape;

	public Tensor(__m256[] _xmm, List<int> _shape) {
		shape = new List<int>(_shape);
		int tmpsize = 1;
		foreach (int x in shape)
			tmpsize *= (int)x;
		Resize(tmpsize);
		Array.Copy(_xmm,xmm,_xmm.Length);
	}

	public Tensor(List<int> _shape) {
		shape = new List<int>(_shape);
		int tmpsize = 1;

		foreach (int x in shape)
			tmpsize *= x;

		Resize(tmpsize,true);
	}

	protected unsafe void Resize(int tmpsize, bool cleanup=false){
		if ( size != tmpsize)
		  {
			size = tmpsize;
			int xmm_size =   (size /8)+((size%8)>0?1:0);
			xmm = new __m256[(int)xmm_size];
			fixed(__m256 *x = xmm){
				pFloats = (float*)x;
			 }
			if (cleanup)
				Array.Fill(xmm,Vector256<float>.Zero);
		  }
	}

	public void CopyTo(Tensor dest){
		dest.Resize(size);
		Array.Copy(xmm,dest.xmm,xmm.Length);
		dest.shape = shape;
	}


	public Tensor(){
        shape = new List<int>();
	}

	~Tensor() {
		shape.Clear();
		size= 0;		
	}


	 public unsafe void load(float[] vec) {
		 Resize(vec.Length);
		 fixed(float *pSrc = vec)
		 {
		 	Buffer.MemoryCopy(pSrc,pFloats,size,size);
		 }
	}

	public unsafe void load(BinaryReader file) {
		xmm[xmm.Length - 1] = Vector256<float>.Zero; //Last one to zero because it can be partially loaded
		float* p2 = pFloats;
		for (int i=0;i< size;++i)
		{
			*p2 =  file.ReadSingle();
			++p2;
		}
	}

	public Tensor(float[] vec, List<int> _shape)  {
		int tmpsize = 1;

		foreach (int x in shape)
			tmpsize *= x;
		Resize(tmpsize);

		load(vec);
	}

	public void save(BinaryWriter file) {
		for (int i=0;i< size;++i)
		{
			file.Write( getElement(i));
		}
	}

	// Get a single element (float value) from the matrix
	 [MethodImpl(MethodImplOptions.AggressiveInlining)] 
	 public unsafe float getElement( int index)  {
	   return *(pFloats+index);
	}

	// Set a single element (float value) into the matrix
	// Caution: This might be an expensive operation if called multiple times. Use setChunk instead
	 [MethodImpl(MethodImplOptions.AggressiveInlining)] 
	 public unsafe void setElement( int index,  float value) {
 	    *(pFloats+index)= value;
	}

	// Sets eight numbers together (__m256).
	 [MethodImpl(MethodImplOptions.AggressiveInlining)]
	 public void setChunk(int index, __m256 chunk) {
		xmm[index] = chunk;
	}

	// Retrieve a chunk from the matrix
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public __m256 getChunk(int index) {
		return xmm[index];
	}
	// Adds two matrices together, mainly for the bias.
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void add(Tensor bias, Tensor o) {
		for (int i = 0; i < bias.xmm.Length; ++i) {
			o.setChunk(i, Avx2.Add(xmm[i], bias.xmm[i]));
		}
	}
	//Subtract two matrices.
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void sub(Tensor a, Tensor o) {
		for (int i = 0; i < xmm.Length; i++) {
			o.xmm[i] =  Avx2.Subtract(xmm[i], a.xmm[i]);
		}
	}

	// Sub that takes a single value instead of an entire matrix
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void sub(float a, Tensor o) {
		__m256 sub_chunk = Vector256.Create(a);
		for (int i = 0; i < xmm.Length; i++) {
			o.xmm[i] = Avx2.Subtract(xmm[i], sub_chunk);
		}
	}

[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void mul(float a, Tensor o) {
		__m256 mul_chunk = Vector256.Create(a);
		for (int i = 0; i < xmm.Length; i++) {
			o.xmm[i] = Avx2.Multiply(xmm[i], mul_chunk);
		}
	}

	// Calculates dot product of two matrices
	// Out is expected to be initialized with its xmm List already resized to the correct length
[MethodImpl(MethodImplOptions.AggressiveInlining)]	
	public void dot_product(int kept_dim, List<float> big_matrix_vec, int big_reserve_size,
		Tensor small, int chunk_range, Tensor o) {
		int out_index = 0;
		for (int small_chunk = 0; small_chunk < small.xmm.Length; small_chunk += chunk_range) {
			for (int big_chunk = 0; big_chunk < xmm.Length; big_chunk += chunk_range) {
				__m256 FMA =  Vector256<float>.Zero;
				for (int partial_index = 0; partial_index < chunk_range; ++partial_index) {
					FMA =  Fma.MultiplyAdd(xmm[big_chunk + partial_index], small.xmm[small_chunk + partial_index], FMA);
				}
				o.setElement(out_index++,NN_utils.hsums(FMA).GetElement<float>(0));
			}
		}
	}

	// Prints the shape.
	public string shape_str()  {
		string _shape_str = "(None, ";

		for (int i = 0; i < shape.Count - 1; i++) {
			_shape_str += (shape[i]) + ", ";
		}
		_shape_str += (shape[shape.Count - 1]) + ")";
		return _shape_str;
	}
	// reshapes the matrix.
	public void reshape(List<int> new_shape) {
		int new_size = 1;

		foreach (int x in new_shape) {
			new_size *= x;
		}
		Debug.Assert(size == new_size);
		shape = new_shape;
	}

	// operator << : Displays contents of matrix
	public override string ToString(){
		string stream = "[";
		float* item = pFloats;
		for (int i=0;i<size;++i)
		{
			stream += *item+" ";
			++item;
		}
		stream += "]";
		return stream;
	}




}

delegate void Activator( Tensor input, Tensor output); 

static class Activators{
	
public static readonly Activator NONE = Activation_Identity;
public static readonly Activator RELU = Activation_ReLU;
public static readonly Activator TANH = Activation_Tanh;
public static readonly Activator SIGMOID = Activation_Sigmoid;
public static readonly Activator SOFTMAX = Activation_Softmax;

//function<void(const Tensor&, Tensor&)> function
public static void Activation_Identity(Tensor input, Tensor output) {
	if (input != output) {
		for (int i = 0; i < output.xmm.Length; ++i)
		{
			output.xmm[i] = input.xmm[i];
		}	
	}
}
public static void Activation_ReLU( Tensor input, Tensor output) {
	__m256 zero = Vector256<float>.Zero;
	for (int i = 0; i < output.xmm.Length; ++i)
	{
		output.xmm[i] = Avx2.Max(input.xmm[i], zero);
	}
}
public static float ReLU_Alpha = 0.04f; //TODO: Find a way to pass it as parameter
public static void Activation_LeakyReLU( Tensor input, Tensor output) {
	__m256 zero = Vector256<float>.Zero;
	 __m256 vAlpha = Vector256.Create(ReLU_Alpha);
	for (int i = 0; i < output.xmm.Length; ++i)
	{
		output.xmm[i] = Avx2.Add(Avx2.Max(input.xmm[i], zero), Avx2.Min(Avx2.Multiply(input.xmm[i], vAlpha), zero));
	}
}

public static void  Activation_Softmax( Tensor input, Tensor output){
	float sum = 0.0f;
	for (int i = 0; i < output.xmm.Length; ++i)
	{
		output.xmm[i] = NN_utils.exp256_ps(input.xmm[i]);
		sum += NN_utils.hsums(output.xmm[i]).GetElement<float>(0);
	}
	sum = 1.0f / sum;
	output.mul(sum, output);
}

public static unsafe void Activation_Tanh( Tensor input, Tensor output) {
	var inPTR = input.pFloats;
	var outPTR = output.pFloats;
	//TODO: convert to Sleef_tanhf8_u10
	for (int i = 0; i < output.size; ++i){
		*outPTR = (float)Math.Tanh( (double)(*inPTR));
		++outPTR;
		++inPTR;
	}
/*
	for (int i = 0; i < output.size; ++i) {
		output.setElement(i,(float)Math.Tanh((double)input.getElement(i)));
	}*/
}
public static void Activation_Sigmoid( Tensor input, Tensor output) {
	__m256 one = Vector256.Create(1.0f);
	for (int i = 0; i < output.xmm.Length; ++i)
	{
		output.xmm[i] = NN_utils.exp256_ps(input.xmm[i]);
		var divisor =  Avx2.Add(output.xmm[i], one);
		output.xmm[i] = Avx2.Divide(output.xmm[i], divisor);
	}
}
}

abstract class Layer  {
	public Layer inputLayer = null;
	public List<Layer> outputLayers = new List<Layer>();

	public Tensor output;
	public Activator activator = null;
	public string name;
	public Layer get(){ return this;}
	public Layer link(Layer _linkLayer)
	{
		Debug.Assert(_linkLayer.output.size > 0);
		Debug.Assert(inputLayer == null); //Cannot connect to multiple inputs
		inputLayer = _linkLayer;
		inputLayer.outputLayers.Add(this);
		//Redimension based on input Weights and Bias
		initialize(inputLayer.output.shape);
		return this; //return this Layer for future connections
	}

	public static Layer operator+(Layer A,Layer _linkLayer)
	{
		return A.link(_linkLayer);
	}

	public int rem=0;

	public Layer(string _name, Activator _activator = null){
  	   name = _name;
		 if (_activator == null)
		 		 activator = Activators.Activation_Identity;
			else activator = _activator;
	 }

	 ~Layer() {}
	//	Calculate output is the function that computes the output of this layer.
	public abstract void calculateOutput(Tensor input_mat);
	public abstract  void predict() ;
	//	Precomute sets up the required matrices and variables required for calculateOutput to work.
	public abstract void initialize(List<int> Dim) ;
	public abstract void precompute() ;
	public abstract void load(BinaryReader file);
	public abstract void save(BinaryWriter file);
	public abstract string getType();
	public abstract int countParams();
	public virtual int summary() {
		int trainableParams = countParams();
		Console.Error.WriteLine(  (name + " (" + getType() + ")").PadRight(28) + output.shape_str().PadRight(26) + trainableParams );
		return trainableParams;
	}
};

class ActivateLayer :  Layer {

	public ActivateLayer(string name, Activator act) : base(name, act) {}
	[MethodImpl(MethodImplOptions.AggressiveInlining)] 
	public  override void calculateOutput(Tensor input_mat){
		activator(input_mat, output);
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)] 
	 public override void predict() {
		Debug.Assert(inputLayer != null);
		calculateOutput(inputLayer.output);
	}
	public override void initialize(List<int> Dim) {
		output = new Tensor(Dim);
	}
	public override void precompute() {
		Debug.Assert(inputLayer != null);
		output = inputLayer.output; //Same Tensor
	}
	public override string getType(){return "ActivateLayer";}
	public override void load(BinaryReader file) {}
	public override void save(BinaryWriter file) {}
	public override int countParams()  { return 0; }
};
class Softmax : ActivateLayer {
	public Softmax(string name = "Softmax") :base(name, Activators.SOFTMAX) {}
	public override string getType()  { return "Softmax"; }
};
class Tanh : ActivateLayer {
	public Tanh(string name = "Tanh") :base(name, Activators.TANH) {}
	public override string getType()  { return "Tanh"; }
};
class ReLU : ActivateLayer {
	public ReLU(string name = "ReLU") :base(name, Activators.RELU) {}
	public override string getType()  { return "ReLU"; }
};
class Sigmoid : ActivateLayer {
	public Sigmoid(string name = "Sigmoid") :base(name, Activators.SIGMOID) {}
	public override string getType()  { return "Sigmoid"; }
};

 class WeightBiasLayer : Layer { //WeightBiasLayer
	protected Tensor weights;
	protected Tensor bias;
	protected int num_of_outputs;
	public	WeightBiasLayer(string name, Activator activator, int _num_of_outputs)
		: base(name, activator) {
 			num_of_outputs=_num_of_outputs;
		}

	~WeightBiasLayer() {}
	//virtual void calculateOutput(Tensor &inputMat) = 0;

	[MethodImpl(MethodImplOptions.AggressiveInlining)] 
	public override void predict() {
		Debug.Assert(inputLayer != null);
		calculateOutput(inputLayer.output);
	}

	public override void load(BinaryReader file) {
		weights.load(file);
		bias.load(file);
	}
	public override void save(BinaryWriter file) {
		weights.save(file);
		bias.save(file);
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public override int countParams(){ return (int)(weights.size + bias.size); }

	public override void precompute() {	}
	public override string getType()  { return "WeightBiasLayer"; }	
	public override void initialize(List<int> Dim) {	}
	public override void calculateOutput(Tensor input_mat){}
};

class Input : Layer {
	public Input(string name, List<int> input_dim) : base(name)//, input_dim(input_dim) 
	{
		output = new Tensor(input_dim);
	}
	public Input(List<int> input_dim) : base("Input")//, input_dim(input_dim) 
	{
		output = new Tensor(input_dim);
	}
	~Input() {}
	public override	void calculateOutput(Tensor input_mat) {
		if (output != input_mat)
		{
			input_mat.CopyTo(output);
		}
	}
	//Somebody took care to update output to the correct inputs....
	[MethodImpl(MethodImplOptions.AggressiveInlining)] 	
	public override void predict() {}
	public override	void initialize(List<int> Dim)  {}//Already done at constructor
	public override	void precompute() {
		//Already done at constructor
	}
	public override	void load(BinaryReader file) {}
	public override	void save(BinaryWriter file) {}
	public override	string getType()  { return "Input"; }
	[MethodImpl(MethodImplOptions.AggressiveInlining)] 
	public override	int countParams()  { return 0; }
	public override	int summary()  { return 0; }
};

class Dense : WeightBiasLayer {

	public Tensor kernelMatrix;
	public Dense(string name, int num_of_outputs, Activator activator)
		: base(name, activator, num_of_outputs) {}
	public Dense(int num_of_outputs, Activator activator = null)
		: base("Dense", activator, num_of_outputs) {}

	~Dense() {}

[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public override	unsafe void calculateOutput(Tensor input_mat) {

		//right now it couldn't be parallelized, 
		fixed (__m256* ori = kernelMatrix.xmm){
			__m256* w = ori;
			for (int n = 0; n < output.xmm.Length; ++n) {
				fixed (__m256* imm = input_mat.xmm)
				{
					float* initData = (float*)imm;
					__m256 accum = Vector256<float>.Zero;
					for (int i = 0; i < input_mat.size; i++) {
						accum = Fma.MultiplyAdd(*w, Vector256.Create((*initData)), accum);
						initData++;
						w++;
					}
					output.xmm[n] = accum;
				}
			}
			output.add(bias, output);
			activator(output, output);
		}
	}

	// Sets up the Dense layer, it takes the shape of the matrix before it to compute its own matrices.
	public override	void initialize(List<int> Dim)  {
		int totalSize = 1;
		foreach (int n in Dim)
			totalSize *= n;
		output = new Tensor(new List<int>{num_of_outputs});
		weights = new Tensor(new List<int>{totalSize, num_of_outputs});
		bias = new Tensor(new List<int>{num_of_outputs});
	}

	public override	unsafe void precompute()  {
		kernelMatrix = new Tensor(new List<int>{ (int)(weights.shape[0] * output.xmm.Length * 8)});
		int CountT = 0;
		for (int n = 0; n < num_of_outputs; n += 8) {
			for (int i = 0; i < inputLayer.output.size; i++) {
				var Tf = i * num_of_outputs + n;
				kernelMatrix.xmm[CountT++] = Avx2.LoadVector256( (weights.pFloats + Tf) );
			}
		}
	}

	public override	string getType() { return "Dense"; }

	}



class Model {
	public List<Input>	inputs=null;
	public List<Layer>	outputs=null;
	public bool Loaded = false;
	private List<Layer>				m_forwardPath = new List<Layer>();  //Predicts all outputs
	private Dictionary<Layer, List<Layer>> m_forwardSingle = new Dictionary<Layer, List<Layer>>(); //For single output predictions

	public Model(List<Input> _inputs, List<Layer> _outputs)
	{
	     inputs = _inputs;
		 outputs = _outputs;
		buildPath();
	}
	public Model(Input _input, Layer value, Layer policy = null)
	{
		inputs = new List<Input>();
		inputs.Add(_input);
		outputs = new List<Layer>();
		outputs.Add(value);
		if (policy != null)
			outputs.Add(policy);
		buildPath();
	}	
	Model() {
	}
	~Model() {
		if (inputs!=null)
			inputs.Clear();
		if (outputs!=null)
			outputs.Clear();
		if (m_forwardPath!= null)	
			m_forwardPath.Clear();
	}
	public void predict(){
	if (!Loaded)
		throw new Exception("Model weights not loaded");
	foreach (var m in m_forwardPath) {
		m.predict();
	}
}
	public void predict(Tensor i){
	if (!Loaded)
		throw new Exception("Model weights not loaded");
	i.CopyTo(inputs[0].output);
	foreach (var m in m_forwardPath) {
		m.predict();
	}
}


	public void predictSingleOutput(Layer output) {
	if (!Loaded)
		throw new Exception("Model weights not loaded");
	foreach (var m in m_forwardSingle[output.get()]) {
		m.predict();
	}
}
	public void summary() {
	int trainableParams = 0;
	const string line = "_________________________________________________________________";
	const string line2 = "=================================================================";
	Console.Error.WriteLine( line );
	Console.Error.WriteLine("Layer (type)                Output Shape              Param #    " );
	for (int i = 1; i < m_forwardPath.Count; ++i)
	{
		Console.Error.WriteLine((i == 1 ? line2 : line)); //Skip 1st layer, input
		trainableParams += m_forwardPath[i].summary();
	}
	Console.Error.WriteLine( line2 );
	Console.Error.WriteLine( "Total params : " + trainableParams.ToString("N0") );
	Console.Error.WriteLine( "Trainable params : " + trainableParams.ToString("N0") );
	Console.Error.WriteLine( "Non - trainable params : 0" );
	Console.Error.WriteLine( line );
}
	

	public void loadWeights(string f){
	using (BinaryReader os = new BinaryReader(File.Open(f, FileMode.Open)))
	{
		foreach (var m in m_forwardPath) {
			m.load(os);
		}
		foreach (var m in m_forwardPath) {
			m.precompute();
		}
		Loaded = true;
	}
}
	public void saveWeights(string f) {
		using (BinaryWriter os = new BinaryWriter(File.Open(f, FileMode.Create)))
		{
			foreach (var m in m_forwardPath) {
				m.save(os);
			}
		}
	}
	private void buildPath(){
			m_forwardPath.Clear();
			Dictionary<Layer, int> deps = new Dictionary<Layer, int>();

			foreach (var l in inputs)
			{
				m_forwardPath.Add(l.get());
			}

			for (int i = 0; i < m_forwardPath.Count; i++)
			{
				var layer = m_forwardPath[i];
				foreach (var wp in layer.outputLayers)
				{
					var next = wp.get();
					const int n = 1;/*next.in;
					if (n > 1)
						deps[layer]++;*/

					if (n == 1 || n == deps[next])
						m_forwardPath.Add(next);
				}
			}
			//Create path for single outputs. I.e. one path for policy and another for value
			m_forwardSingle.Clear();
			deps.Clear();
			foreach (var O in outputs) {
				Layer layer = O.get();
				if (m_forwardSingle.ContainsKey(layer))
					m_forwardSingle[layer].Clear();
				else m_forwardSingle.Add(layer, new List<Layer>());
				if (deps.ContainsKey(layer))
					deps[layer]=0;
				else deps.Add(layer, 0);

				while (layer != null && deps[layer] == 0) {
					m_forwardSingle[layer].Insert(0, layer);
					deps[layer]++;
					layer = layer.inputLayer;
					if (layer == null) 
						break;
					if (!m_forwardSingle.ContainsKey(layer))
						m_forwardSingle.Add(layer, new List<Layer>());
					if (!deps.ContainsKey(layer))
					    deps.Add(layer, 0);					
				}
			}

	}

};


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

class Example {
	double[][][] input;
	int label;
	
	public Example(double[][] input, int label){
		double[][][] in = new double[1][input.length][input[0].length];
		in[0] = input;
		this.input = in;
		this.label = label;
	}
}

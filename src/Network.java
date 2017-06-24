import java.util.Arrays;
import java.util.Random;

class Network {

	final int imageSize = 32;
	final int numClasses = 5;
	Layer[] layers = new Layer[7];

	// convconfig [numFilters, kernelSize, poolSize] fullconfig [neuronsOut]
	public Network(int[] conv1Config, int[] conv2Config, int[] conv3Config,
			int[] fullCon1Config, int[] fullCon2Config) {
		layers[0] = new Convolution(imageSize, conv1Config[0], conv1Config[1],
				conv1Config[2]);

		int conv2NeuronsIn = (int) ((double) (imageSize - conv1Config[1] + 1) / (double) conv1Config[2]);
		layers[1] = new Convolution(conv2NeuronsIn, conv2Config[0],
				conv2Config[1], conv2Config[2]);

		int conv3NeuronsIn = (int) ((double) (conv2NeuronsIn - conv2Config[1] + 1) / (double) conv2Config[2]);
		layers[2] = new Convolution(conv3NeuronsIn, conv3Config[0],
				conv3Config[1], conv3Config[2]);

		int sizeOfPool = (int) ((double) (conv3NeuronsIn - conv3Config[1] + 1) / (double) conv3Config[2]);
		layers[3] = new Flatten(conv3Config[0], sizeOfPool);

		int fullConNeuronsIn = conv3Config[0] * sizeOfPool * sizeOfPool;
		layers[4] = new FullyConnected(fullConNeuronsIn, fullCon1Config[0]);
		layers[5] = new FullyConnected(fullCon1Config[0], fullCon2Config[0]);
		layers[6] = new Softmax(fullCon2Config[0], numClasses);

	}

	public double[] feedforward(double[][][] input) {
		// double[][][] in = new double[1][input.length][input[0].length];
		// in[0] = input;

		double[][][] conv1 = ((Convolution) layers[0])
				.feedforward(input, false);
		double[][][] conv2 = ((Convolution) layers[1])
				.feedforward(conv1, false);
		double[][][] conv3 = ((Convolution) layers[2])
				.feedforward(conv2, false);
		double[] flatten = ((Flatten) layers[3]).feedforward(conv3);
		double[] fullCon1 = ((FullyConnected) layers[4]).feedforward(flatten,
				false);
		double[] fullCon2 = ((FullyConnected) layers[5]).feedforward(fullCon1,
				false);
		double[] softmax = ((Softmax) layers[6]).feedforward(fullCon2, false);
		return softmax;
	}

	public void train(int epochs, double lambda, double momentum,
			Example[] dataSet, int miniBatchSize) {

		// shuffle dataSet
		dataSet = shuffle(dataSet);
		int tenPec = (int) (dataSet.length * 0.1);
		// testSet = 10% of dataSet
		Example[] testSet = Arrays.copyOfRange(dataSet, 0, tenPec);
		// validationSet = 10% of dataSet
		Example[] validationSet = Arrays.copyOfRange(dataSet, 0, // tenPec,
				tenPec * 2);
//		Example[][] valBatch = new Example[2][validationSet.length / 2];
//		for (int v = 0; v < valBatch.length; v++) {
//			for (int e = 0; e < valBatch[0].length; e++) {
//				valBatch[v][e] = validationSet[v * valBatch[0].length + e];
//			}
//		}
		// trainingSet = 80% of dataSet
		Example[] trainingSet = Arrays.copyOfRange(dataSet, tenPec * 2,
				dataSet.length);
		int trainingSize = trainingSet.length;

		for (int epoch = 0; epoch < epochs; epoch++) {
			System.out.println("Epoch: " + epoch);
			// shuffle trainingSet
			trainingSet = shuffle(trainingSet);
			// seperate dataSets into batches
			int numBatches = trainingSize / miniBatchSize;
			// Example[][] batches = new Example[numBatches][miniBatchSize];
			for (int batch = 0; batch < numBatches; batch++) {
				System.out.println("  Batch: " + batch);
//				long starttime = System.currentTimeMillis();
				
				for (int ex = 0; ex < miniBatchSize; ex++) {
					// batches[batch][ex] = trainingSet[batch*miniBatchSize+ex];
					Example example = trainingSet[batch * miniBatchSize + ex];

					double[][][] conv1 = ((Convolution) layers[0]).feedforward(
							example.input, true);
					double[][][] conv2 = ((Convolution) layers[1]).feedforward(
							conv1, true);
					double[][][] conv3 = ((Convolution) layers[2]).feedforward(
							conv2, true);
					double[] flatten = ((Flatten) layers[3]).feedforward(conv3);
					double[] fullCon1 = ((FullyConnected) layers[4])
							.feedforward(flatten, true);
					double[] fullCon2 = ((FullyConnected) layers[5])
							.feedforward(fullCon1, true);
					double[] softmax = ((Softmax) layers[6]).feedforward(
							fullCon2, true);

					// calc network deltas
					double[] deltas = new double[numClasses];
					for (int d = 0; d < numClasses; d++) {
						if (d == example.label) {
							deltas[d] = softmax[d] - 1;
						} else {
							deltas[d] = softmax[d];
						}
					}

					if (ex == 0) {
						System.out.println("    Error: "
								+ cost(trainingSize, softmax));
					}

					// backprop deltas through network
					double[] softmaxDeltas = ((Softmax) layers[6]).backprop(
							deltas, momentum);
					double[] fullCon2Deltas = ((FullyConnected) layers[5])
							.backprop(softmaxDeltas, momentum);
					double[] fullCon1Deltas = ((FullyConnected) layers[4])
							.backprop(fullCon2Deltas, momentum);
					double[][][] flattenDeltas = ((Flatten) layers[3])
							.backprop(fullCon1Deltas);
					double[][][] conv3Deltas = ((Convolution) layers[2])
							.backprop(flattenDeltas, momentum);
					double[][][] conv2Deltas = ((Convolution) layers[1])
							.backprop(conv3Deltas, momentum);
					double[][][] conv1Deltas = ((Convolution) layers[0])
							.backprop(conv2Deltas, momentum);
				}

				// update weights in each layer
				for (int l = 0; l < layers.length; l++) {
					layers[l].updateWeights(miniBatchSize, lambda,
							trainingSize);
				}
			}

			// evalute against validationSet
			double validationScore = evaluate(validationSet);
			System.out.println("Epoch: " + epoch + " validation score: "
					+ validationScore);
		}

		// evalute against testSet
		double testScore = evaluate(testSet);
		System.out.println("test score: " + testScore);
	}

	// initalize weights for each layer
	public void initweights() {
		for (int l = 0; l < layers.length; l++) {
			layers[l].initweights();
		}
	}

	// evaluates the network against test data
	public double evaluate(Example[] testSet) {
		int c = testSet.length;
		int correct = 0;
		for (int e = 0; e < c; e++) {
			double[] acts = feedforward(testSet[e].input);
			int result = 0;
			for (int i = 0; i < numClasses; i++) {
				if (acts[i] > acts[result]) {
					result = i;
				}
			}
			if (result == testSet[e].label) {
				correct++;
			}
		}
		return (double) correct / (double) testSet.length;
	}

	public double cost(int trainingsize, double[] acts) {
		double sum = 0;

		for (int x = 0; x < numClasses; x++) {
			sum = sum + Math.log(acts[x]);
		}
		return -(sum / (double) trainingsize) * 1000;
	}

	// shuffle dataset array
	public Example[] shuffle(Example[] dataset) {
		int c = dataset.length;
		Random r = new Random();
		for (int i = 0; i < c; i++) {
			Example ex1 = dataset[i];
			int index = r.nextInt(c);
			Example ex2 = dataset[index];

			dataset[i] = ex2;
			dataset[index] = ex1;
		}

		return dataset;
	}

}

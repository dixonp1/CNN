class Softmax implements Layer{
    int neuronsIN;
    int neuronsOUT;
    
    double[] biases;
    double[][] weights;
    
    double[] bGradients;
    double[][] wGradients;
    
    double[] bDeltas;
    double[][] wDeltas;

    double[] lastInput;
    final double smoothing = 0.0000001;
    
    public Softmax(int neuronsIN, int neuronsOUT){
        this.neuronsIN = neuronsIN;
        this.neuronsOUT = neuronsOUT;
        
        biases = new double[neuronsOUT];
        weights = new double[neuronsOUT][neuronsIN];
        
        bGradients = new double[neuronsOUT];
        wGradients = new double[neuronsOUT][neuronsIN];
        
        bDeltas = new double[neuronsOUT];
        wDeltas = new double[neuronsOUT][neuronsIN];
        
        lastInput = new double[neuronsIN];
    }

    //feedforward input
    public double[] feedforward(double[] input, boolean training){
        double[] z = new double[neuronsOUT];
        for(int k=0; k<neuronsOUT; k++){
            z[k] = biases[k];
            for(int j=0; j<neuronsIN; j++){
                z[k] += weights[k][j] * input[j];
            }
        }
        
        if(training){
            lastInput = input;
        }
        
        return softmax(z);
    }

    //initalize weights
    public void initweights(){}

    //update weights with gradients
    public void updateWeights(int miniBatchSize, double lambda, int trainingSize){
    	double reg = 1 - lambda / (double)trainingSize;
        for(int k=0; k<neuronsOUT; k++){
            biases[k] = biases[k] + bDeltas[k]/(double)miniBatchSize;
            for(int j=0; j<neuronsIN; j++){
                weights[k][j] = reg * weights[k][j] + wDeltas[k][j]/(double)miniBatchSize;
            }
        }
    }

    public double[] backprop(double[] nextDeltas, double momentum){
        //if first iteration in minibatch, clear old gradients
//        if(mbits == 0){
//        	for(int k=0; k<neuronsOUT; k++){
//                bGradients[k] = momentum * eta * bGradients[k];
//                for(int j=0; j<neuronsIN; j++){
//                    wGradients[k][j] = momentum * eta * wGradients[k][j];	
//                }
//        	}
//        }
        
        //calc gradients for weights
    	for(int k=0; k<neuronsOUT; k++){
            bGradients[k] = momentum * bGradients[k] + (1-momentum)*
            		((nextDeltas[k] * nextDeltas[k]));
            double bUpdate = -(Math.sqrt(bDeltas[k] + smoothing)/Math.sqrt(bGradients[k] + smoothing)) 
            		* nextDeltas[k];            
            bDeltas[k] = momentum * bDeltas[k] + (1-momentum) * (bUpdate*bUpdate);
            
            for(int j=0; j<neuronsIN; j++){
                wGradients[k][j] = momentum*wGradients[k][j] + (1-momentum) * 
                		((nextDeltas[k] * lastInput[j]) * (nextDeltas[k] * lastInput[j]));
                double wUpdate = -(Math.sqrt(wDeltas[k][j] + smoothing)/Math.sqrt(wGradients[k][j] + smoothing))
                		* (nextDeltas[k] * lastInput[j]);
                wDeltas[k][j] = momentum * wDeltas[k][j] + (1-momentum) * (wUpdate*wUpdate);
            }
        }
 
        //calc deltas for previous layer
        double[] prevDeltas = new double[neuronsIN];
        for(int k=0; k<neuronsOUT; k++){
            for(int j=0; j<neuronsIN; j++){
                prevDeltas[j] = prevDeltas[j] + (nextDeltas[k] * weights[k][j]);
            }
        }
        return prevDeltas;
    }
    

    public double[] softmax(double[] z){
        double[] acts = new double[neuronsOUT];
        double sum = 0;
        double stabilize = z[0];
        
        for(int i=0; i<z.length; i++){
        	if(z[i] > stabilize){
        		stabilize = z[i];
        	}
        }
        
        for(int k=0; k<neuronsOUT; k++){
            sum = sum + Math.pow(Math.E, z[k] - stabilize);
        }
        for(int k=0; k<neuronsOUT; k++){
            acts[k] = Math.pow(Math.E, z[k] - stabilize) / sum;
        }
        
        return acts;
    }
}


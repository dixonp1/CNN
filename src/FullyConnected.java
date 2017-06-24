import java.util.Random;


class FullyConnected implements Layer{
	 int neuronsIN;
	 int neuronsOUT;
	    
	 double[] biases;
	 double[][] weights;
	    
	 double[] bGradients;
	 double[][] wGradients;
	 
	 double[] bDeltas;
	 double[][] wDeltas;

	 double[] lastInput;
	 double[] lastAct;
	 
	 final double smoothing = 0.0000001;
	 final double dropout = 0.5; 
	 double[] dropmask;
    
    public FullyConnected(int neuronsIN, int neuronsOUT){
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
       	double[] acts = new double[neuronsOUT];
        for(int k=0; k<neuronsOUT; k++){
            double z = biases[k];
            for(int j=0; j<neuronsIN; j++){
                z += weights[k][j] * input[j];
                acts[k] = ReLU(z);
            }
        }
        
        if(training){
            lastInput = input;
            lastAct = acts;
            
            if(dropmask == null){
            	dropmask = new double[neuronsOUT];
            	Random r = new Random();
            	for(int i=0; i<neuronsOUT; i++){
            		if(r.nextInt(100) <= dropout*100){
            			dropmask[i] = 1/dropout;
            			acts[i] *= dropmask[i];
            		}
            	}
            }
        }
        return acts;
    }

    //initalize weights
    public void initweights(){
    	Random r = new Random();
        
        for(int k=0; k<neuronsOUT; k++){
            biases[k] = r.nextGaussian() * 1/Math.sqrt((double)neuronsIN);
            for(int j=0; j<neuronsIN; j++){
                weights[k][j] = r.nextGaussian() * 1/Math.sqrt((double)neuronsIN);
            }
        }
    }

    //update weights
    public void updateWeights(int miniBatchSize, double lambda, int trainingSize){
    	double reg = 1 - lambda / (double)trainingSize;
        for(int k=0; k<neuronsOUT; k++){
            biases[k] = biases[k] + bDeltas[k]/(double)miniBatchSize;
            for(int j=0; j<neuronsIN; j++){
                weights[k][j] = reg * weights[k][j] + wDeltas[k][j]/(double)miniBatchSize;
            }
        }
        dropmask = null;
    }

    public double[] backprop(double[] nextDeltas, double momentum){
        //reset gradients if first iteration in minibatch
//    	if(mbits == 0){
//    		for(int k=0; k<neuronsOUT; k++){
//                bGradients[k] = momentum * eta * bGradients[k];
//                for(int j=0; j<neuronsIN; j++){
//                    wGradients[k][j] = momentum * eta * wGradients[k][j];	
//                }
//        	}
//        }
        
        //calc my deltas and gradients
        double[] deltas = new double[neuronsOUT];
        for(int k=0; k<neuronsOUT; k++){
            deltas[k] = nextDeltas[k] * ReLUPrime(lastAct[k]) * dropmask[k];
            bGradients[k] = momentum * bGradients[k] + (1-momentum)*((deltas[k] * deltas[k]));
            double bUpdate = -(Math.sqrt(bDeltas[k] + smoothing)/Math.sqrt(bGradients[k] + smoothing)) * deltas[k];            
            bDeltas[k] = momentum * bDeltas[k] + (1-momentum) * (bUpdate*bUpdate);
            
            for(int j=0; j<neuronsIN; j++){
                wGradients[k][j] = momentum*wGradients[k][j] + (1-momentum) * 
                		((deltas[k] * lastInput[j]) * (deltas[k] * lastInput[j]));
                double wUpdate = -(Math.sqrt(wDeltas[k][j] + smoothing)/Math.sqrt(wGradients[k][j] + smoothing))
                		* (deltas[k] * lastInput[j]);
                wDeltas[k][j] = momentum * wDeltas[k][j] + (1-momentum) * (wUpdate*wUpdate);
            }
        }
        
        //calc deltas for previous layer
        double[] prevDeltas = new double[neuronsIN];
        for(int j=0; j<neuronsIN; j++){
            for(int k=0; k<neuronsOUT; k++){
                prevDeltas[j] = prevDeltas[j] + (deltas[k] * weights[k][j]);
            }
        }
        return prevDeltas;
    }

    //activation function
    public double ReLU(double x){
        double y = 0.0;
        if(x>y){
            y=x;
        }
        return y;
    }
    
    //derivitive of activation function
    public double ReLUPrime(double x){
        double y = 0.0;
        if(x>y){
            y = 1.0;
        }
        return y;
    }

}

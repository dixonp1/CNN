import java.util.Random;

class Convolution implements Layer{
    int neuronsIN;
    int numFilters;
    int kernelSize;
    int poolSize;
    int mapsize;
    
    double[] biases;
    double[][][] weights;
    
    double[] bGradients;
    double[][][] wGradients;
    
    double[] bDeltas;
    double[][][] wDeltas;
    
    double[][][] lastInput;
    double[][][] lastFilters;
    int[][][] maxpoolloc;
    final double smoothing = 0.0000001;
    
    public Convolution(int neuronsIN, int numFilters, int kernelSize, int poolSize){
        this.neuronsIN = neuronsIN;
        this.numFilters = numFilters;
        this.kernelSize = kernelSize;
        this.poolSize = poolSize;
        mapsize = neuronsIN - kernelSize + 1;
        
        biases = new double[numFilters];
        weights = new double[numFilters][kernelSize][kernelSize];
        
        bGradients = new double[numFilters];
        wGradients = new double[numFilters][kernelSize][kernelSize];
        
        bDeltas = new double[numFilters];
        wDeltas = new double[numFilters][kernelSize][kernelSize];
        
        lastFilters = new double[numFilters][mapsize][mapsize];
        maxpoolloc = new int[numFilters][mapsize/poolSize][mapsize/poolSize];
    }
    
    //feed input forward through layer
    public double[][][] feedforward(double[][][]input, boolean training){
        double[][][] filters = new double[numFilters][mapsize][mapsize];
        double[][][] poollayers = new double[numFilters][mapsize/poolSize][mapsize/poolSize];
        int numImages = input.length;
        
        //if training store pool locations and last input
        if(training){
            lastInput = input;
        }
        
        for(int f=0; f<numFilters; f++){
            for(int k=0; k<mapsize; k++){
                for(int j=0; j<mapsize; j++){
                    double z = biases[f];
                    for(int img=0; img<numImages; img++){
                        for(int b=0; b<kernelSize; b++){
                            for(int a=0; a<kernelSize; a++){
                                z += weights[f][b][a] * input[img][k+b][j+a];
                            }
                        }
                    }
                    filters[f][k][j] = ReLU(z);
                }
            }

            /**********************
                POOLING LAYER
            **********************/
            int sizeofpool = mapsize/poolSize;
            for(int k=0; k<sizeofpool; k++){
                for(int j=0; j<sizeofpool; j++){
                	poollayers[f][k][j] = filters[f][2 * k][2 * j];

                    if(training){
                        maxpoolloc[f][k][j] = (2 * k) * mapsize + (2 * j);
                    }
                        
                    for(int b=0; b<poolSize; b++){
                        for(int a=0; a<poolSize; a++){
                            if(filters[f][b + 2 * k][a + 2 * j] > poollayers[f][k][j]){
                                poollayers[f][k][j] = filters[f][b + 2 * k][a + 2 * j];
                                if(training){
                                    maxpoolloc[f][k][j] = (b + 2 * k) * mapsize + (a + 2 * j);
                                }
                            }
                        }
                    }
                }
            }
        }
        if(training){
            lastFilters = filters;
        }
        return poollayers;
    }

    public void initweights(){
    	Random r = new Random();
        
        for(int f=0; f<numFilters; f++){
            biases[f] = r.nextGaussian() * 1/Math.sqrt((double)neuronsIN * neuronsIN);
            for(int k=0; k<kernelSize; k++){
                for(int j=0; j<kernelSize; j++){
                    weights[f][k][j] = r.nextGaussian() * 1/Math.sqrt((double)neuronsIN * neuronsIN);
                }
            }
        }
    
        /*
        biases = [1.0, 1.0]
        weights = [[[3.0,-1.0],
                    [-7.0,2.0]],
                    [[-1.0,0.5],
                     [0.4,-2.0]]]
        */
        
    }

    //update weights
	public void updateWeights(int miniBatchSize, double lambda, int trainingSize){
        double reg = 1.0 - lambda / (double)trainingSize;
        
        for(int f=0; f<numFilters; f++){
            biases[f] = biases[f] + bDeltas[f]/(double)miniBatchSize;
            for(int b=0; b<kernelSize; b++){
                for(int a=0; a<kernelSize; a++){
                    weights[f][b][a] = reg * weights[f][b][a] + 
                    		wDeltas[f][b][a]/(double)miniBatchSize;
                }
            }
        }
    }

    
    public double[][][] backprop(double[][][] nextDeltas, double momentum){
        //if first iteration in minibatch overwrite old gradients
//        if(mbits == 0){
//            for(int f=0; f<numFilters; f++){
//            	bGradients[f] = momentum * eta * bGradients[f];
//            	for(int b=0; b<kernelSize; b++){
//            		for(int a=0; a<kernelSize; a++){
//            			wGradients[f][b][a] = momentum * eta * wGradients[f][b][a];
//            		}
//            	}
//            }
//        }

        //build feature map deltas
        double[][][] fdeltas = new double[numFilters][mapsize][mapsize];
        int numImages = lastInput.length;
        for(int f=0; f<numFilters; f++){

            //route gradients through poollayers
            //and calc my deltas
        	int sizeofpool = mapsize/poolSize;
            for(int k=0; k<sizeofpool; k++){
                for(int j=0; j<sizeofpool; j++){
                    int loc = maxpoolloc[f][k][j];
                    int x = loc % mapsize;
                    int y = (loc-x)/(mapsize);
                    fdeltas[f][y][x] = nextDeltas[f][k][j] * ReLUPrime(lastFilters[f][y][x]);
                }
            }
        
            //calc my gradients
            //biases
            for(int k=0; k<mapsize; k++){
                for(int j=0; j<mapsize; j++){
                    bGradients[f] = bGradients[f] + fdeltas[f][k][j];
                    bGradients[f] = momentum * bGradients[f] + (1-momentum)
                    		*((fdeltas[f][k][j] * fdeltas[f][k][j]));
                    double bUpdate = -(Math.sqrt(bDeltas[f] + smoothing)/Math.sqrt(bGradients[f] + smoothing)) 
                    		* fdeltas[f][k][j];            
                    bDeltas[f] = momentum * bDeltas[f] + (1-momentum) * (bUpdate*bUpdate);
                }
            }
            
            //weights
            for(int b=0; b<kernelSize; b++){
                for(int a=0; a<kernelSize; a++){
                    for(int img=0; img<numImages; img++){
                        for(int k=0; k<mapsize; k++){
                            for(int j=0; j<mapsize; j++){
                                wGradients[f][b][a] = momentum*wGradients[f][b][a] + 
                                		(1-momentum) * ((fdeltas[f][b][a] * lastInput[img][k+b][j+a]) 
                                				* (fdeltas[f][k][j] * lastInput[img][k+b][j+a]));
                                double wUpdate = -(Math.sqrt(wDeltas[f][b][a] + smoothing)/
                                		Math.sqrt(wGradients[f][b][a] + smoothing)) * (fdeltas[f][k][j] 
                                				* lastInput[img][k+b][j+a]);
                                wDeltas[f][b][a] = momentum * wDeltas[f][b][a] + (1-momentum) 
                                		* (wUpdate*wUpdate);
                            }
                        }
                    }
                }
            }
        }
        
        //fdeltas[0] = [[-1.0,1.0],[2.0,0.0]]
        //propagate error back
        double[][][] prevDeltas = new double[numImages][neuronsIN][neuronsIN];
        for(int i=0; i<numImages; i++){
        	for(int f=0; f<numFilters; f++){
        		for(int k=0; k<neuronsIN; k++){
        			for(int j=0; j<neuronsIN; j++){
        				for(int b=0; b<kernelSize; b++){
        					for(int a=0; a<kernelSize; a++){
        						if ((k-b >= 0 && j-a >= 0) && (k-b < mapsize && j-a < mapsize)){
                                	prevDeltas[i][k][j] += fdeltas[f][k-b][j-a] * weights[f][b][a];
                            	}
                        	}
                    	}
                	}
            	}
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
 


class Flatten implements Layer{
    int numPools;
    int sizeOfPool;
    
    public Flatten(int numPools, int sizeOfPool){
        this.numPools = numPools;
        this.sizeOfPool = sizeOfPool;
    }
    
    //flatten 3D array input into 1D array and feeds forward
    public double[] feedforward(double[][][] input){
        double[] out = new double[numPools * sizeOfPool * sizeOfPool];
        for(int pool=0; pool<numPools; pool++){
            for(int b=0; b<sizeOfPool; b++){
                for(int a=0; a<sizeOfPool; a++){
                    int i = pool * sizeOfPool * sizeOfPool + b * sizeOfPool + a;
                    out[i] = input[pool][b][a];
                }
            }
        }
        return out;
    }
    
    //expands 3D array into a 1D array and propagates it backwards
    public double[][][] backprop(double[] deltas){
        double[][][] pools = new double[numPools][sizeOfPool][sizeOfPool];
        for(int pool=0; pool<numPools; pool++){
            for(int b=0; b<sizeOfPool; b++){
                for(int a=0; a<sizeOfPool; a++){
                    int i = pool * sizeOfPool * sizeOfPool + b * sizeOfPool + a;
                    pools[pool][b][a] = deltas[i];
                }
            }
        }
        return pools;
    }

	public void initweights() {}

	public void updateWeights(int miniBatchSize, double lambda,
			int trainingSize) {}

}

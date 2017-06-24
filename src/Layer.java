
public interface Layer {

    public void initweights();
    public void updateWeights(int miniBatchSize, double lambda, int trainingSize);
}

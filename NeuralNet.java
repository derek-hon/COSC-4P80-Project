package neuralnet;

import java.util.Random;

public class NeuralNet implements Comparable<NeuralNet> {

    double[][][] weights;
    double[][] activations;
    int[] layers;
    double error;
    Random random;
    
    public NeuralNet(int[] layers, Random random) {
        this.layers = layers;
        this.random = random;
        weights = new double[layers.length - 1][0][0];
        activations = new double[layers.length][0];

        for (int i = 0; i < 2; i++) {
            weights[i] = new double[layers[i]][layers[i + 1]];
            activations[i] = new double[layers[i]];
        }
        activations[layers.length - 1] = new double[layers[layers.length - 1]];
        randomizeWeights();
    }

    public double propogateForwardsWithError(double[] inputs, int classification) {
        propogateForwards(inputs);
        error = calculateError(classification);
        return error;
    }

    private void propogateForwards(double[] inputs) {
        System.arraycopy(inputs, 0, activations[0], 0, inputs.length);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < layers[i + 1]; j++) {
                activations[i + 1][j] = 0.0;
                for (int k = 0; k < layers[i]; k++) {
                    activations[i + 1][j] += activations[i][k] * weights[i][k][j];
                }
                activations[i + 1][j] = activationFunction(activations[i + 1][j]);
            }
        }
    }

    private double calculateError(int classification) {
        double err = 0;
        for (int i = 0; i < layers[layers.length - 1]; i++) {
            if (i == classification) {
                err += Math.pow(activations[layers.length - 1][i] - 1, 2);
            } else {
                err += Math.pow(activations[layers.length - 1][i], 2);
            }
        }
        return err / layers[layers.length - 1];
    }

    public double[] flattenWeights() {
        int height = layers.length;
        int width = weights[0].length;
        int depth = weights[0][0].length;
        int size = height * width * depth;

        double[] flattenedWeights = new double[size];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < depth; k++) {
                    flattenedWeights[i + width * (j + depth * k)] = weights[i][j][k];
                }
            }
        }
        return flattenedWeights;
    }

    public void unflattenWeights(double[] flattenedWeights) {
        int height = layers.length;
        int width = weights[0].length;
        int depth = weights[0][0].length;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < depth; k++) {
                    weights[i][j][k] = flattenedWeights[i + width * (j + depth * k)];
                }
            }
        }
    }

    private void randomizeWeights() {
        Random r = new Random();
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j][k] = -0.5 + (1.0) * random.nextDouble();
                }
            }
        }
    }

    private double activationFunction(double x) {
        return relu(x);
    }

    private double relu(double x) {
        return x > 0 ? x : 0;
    }

    private double sigmoid(double x) {
        return (1 / (1 + Math.pow(Math.E, (-1 * x))));
    }
    
    @Override
    public int compareTo(NeuralNet o) {
        return (int) (this.error - o.error);
    }

}

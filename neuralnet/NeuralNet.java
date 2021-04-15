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

        for (int i = 0; i < layers.length - 1; i++) {
            weights[i] = new double[layers[i]][layers[i + 1]];
            activations[i] = new double[layers[i]];
        }
        activations[layers.length - 1] = new double[layers[layers.length - 1]];
        randomizeWeights();
    }

    public int propogateForwardsWithError(double[] inputs, int classification) {
        int classified = propogateForwards(inputs);
        error = calculateError(classification);
        return classified;
    }

    public int propogateForwards(double[] inputs) {
        System.arraycopy(inputs, 0, activations[0], 0, inputs.length);
        for (int i = 0; i < layers.length - 1; i++) {
            for (int j = 0; j < layers[i]; j++) {
                for (int k = 0; k < layers[i + 1]; k++) {
                    activations[i + 1][k] = activations[i][j] * weights[i][j][k];
                }
            }
        }
        double max = 0;
        int maxIndex = 0;
        for (int i = 0; i < activations[activations.length - 1].length; i++) {
            if (activations[activations.length - 1][i] > max) {
                max = activations[activations.length - 1][i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private double calculateError(int classification) {
        double err = 0;
        for (int i = 0; i < layers[layers.length - 1]; i++) {
            if (i == classification) {
                err += Math.pow(activations[layers.length - 1][i] - 1, 2);
            } else {
                err += Math.pow(activations[layers.length - 1][i] * 2, 2);
            }
        }
        return err;// / layers[layers.length - 1];
    }

    public double[] flattenWeights() {
        int size = 0;
        for (int i = 0; i < layers.length - 1; i++) {
            size += layers[i] * layers[i + 1];
        }

        int width = 0;
        double[] flattenedWeights = new double[size];
        for (int i = 0; i < layers.length - 1; i++) {
            for (int j = 0; j < layers[i]; j++) {
                for (int k = 0; k < layers[i + 1]; k++) {
                    flattenedWeights[width + j * layers[i + 1] + k] = weights[i][j][k];
                }
            }
            width += layers[i] * layers[i + 1];
        }
        return flattenedWeights;
    }

    public void unflattenWeights(double[] flattenedWeights) {
        int width = 0;
        for (int i = 0; i < layers.length - 1; i++) {
            for (int j = 0; j < layers[i]; j++) {
                for (int k = 0; k < layers[i + 1]; k++) {
                    weights[i][j][k] = flattenedWeights[width + j * layers[i + 1] + k];
                }
            }
            width += layers[i] * layers[i + 1];
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

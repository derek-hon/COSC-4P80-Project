package neuralnet;

import java.util.List;
import java.util.Random;

public class NeuralNet implements Comparable<NeuralNet> {

    double[][][] weights;
    double[][] activations;
    int[] layers;
    double fitness;
    Random random;

    public NeuralNet() {

    }

    public NeuralNet(int[] layers, Random random) {
        this.layers = layers;
        this.random = new Random();
        weights = new double[layers.length - 1][0][0];
        activations = new double[layers.length][0];

        for (int i = 0; i < layers.length - 1; i++) {
            weights[i] = new double[layers[i]][layers[i + 1]];
            activations[i] = new double[layers[i]];
        }
        activations[layers.length - 1] = new double[layers[layers.length - 1]];
        randomizeWeights();
    }

    public int propogateForwards(double[] inputs) {
        activations[0] = inputs;
        for (int i = 0; i < layers.length - 1; i++) {
            for (int k = 0; k < layers[i + 1]; k++) {
                double tempActivation = 0;
                for (int j = 0; j < layers[i]; j++) {
                    tempActivation += activations[i][j] * weights[i][j][k];
                }
                activations[i + 1][k] = activationFunction(tempActivation);
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

    public double updateFitness(List<InputInstance> digits) {
        int count = 0;
        int correct = 0;
        for (InputInstance digit : digits) {
            if (digit.classification == propogateForwards(digit.values)) {
                correct++;
            }
            count++;
        }
        fitness = (double) correct / count;
        return fitness;
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
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j][k] = -0.5 + (1.0) * random.nextDouble();
                }
            }
        }
    }

    @Override
    public NeuralNet clone() {
        NeuralNet temp = new NeuralNet();
        temp.layers = ArrayCopy1d(this.layers);
        temp.random = this.random;
        temp.weights = ArrayCopy3d(this.weights);
        temp.activations = ArrayCopy2d(this.activations);
        temp.fitness = this.fitness;
        return temp;
    }

    private double[][][] ArrayCopy3d(double[][][] a) {
        double[][][] temp = new double[a.length][][];
        for (int i = 0; i < a.length; i++) {
            temp[i] = new double[a[i].length][];
            for (int j = 0; j < a[i].length; j++) {
                temp[i][j] = new double[a[i][j].length];
                for (int k = 0; k < a[i][j].length; k++) {
                    temp[i][j][k] = a[i][j][k];
                }
            }
        }
        return temp;
    }

    private double[][] ArrayCopy2d(double[][] a) {
        double[][] temp = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            temp[i] = new double[a[i].length];
            for (int j = 0; j < a[i].length; j++) {
                temp[i][j] = a[i][j];
            }
        }
        return temp;
    }

    private int[] ArrayCopy1d(int[] a) {
        int[] temp = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            temp[i] = a[i];
        }
        return temp;
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
        return Double.compare(o.fitness, this.fitness);
    }

}

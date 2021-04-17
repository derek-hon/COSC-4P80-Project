package neuralnet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class GA {

    public double crossoverRate;
    public double mutationRate;
    public int popSize;
    public int elitism;
    public int tourneySize;
    private final Random random;
    public List<NeuralNet> population;

    public GA(double crossoverRate, double mutationRate, int popSize, int elitism, int tourneySize, Random random) {
        this.crossoverRate = crossoverRate;
        this.elitism = elitism;
        this.mutationRate = mutationRate;
        this.popSize = popSize;
        this.tourneySize = tourneySize;
        this.random = random;
        initPopulation();
    }

    private void initPopulation() {
        List<NeuralNet> newPop = new ArrayList<>();
        for (int i = 0; i < popSize; i++) {
            NeuralNet nn = new NeuralNet(new int[]{64, 64,32, 10}, random);
            newPop.add(nn);
        }
        population = newPop;
    }

    public void evaluatePopulation(List<InputInstance> trainData) {
        population.forEach(p -> {
            p.updateFitness(trainData);
        });
    }

    public void selectionProcess() {
        List<NeuralNet> newPop = new ArrayList<>();

        Collections.sort(population);

        for (int i = 0; i < elitism; i++) {
            newPop.add(population.get(i).clone());
        }

        while (newPop.size() < popSize - 1) {
            NeuralNet[] chromosomes = tournamentSelection();
            NeuralNet chromosomeA = chromosomes[0];
            NeuralNet chromosomeB = chromosomes[1];

            if (random.nextDouble() < crossoverRate) {
                NeuralNet[] children;
                children = crossover(chromosomeA, chromosomeB);
                chromosomeA = children[0];
                chromosomeB = children[1];
            }

            if (random.nextDouble() < mutationRate) {
                chromosomeA = mutate(chromosomeA);
            }

            if (random.nextDouble() < mutationRate) {
                chromosomeB = mutate(chromosomeB);
            }
            newPop.add(chromosomeA);
            newPop.add(chromosomeB);
        }

        if (newPop.size() < popSize) {
            NeuralNet[] chromosomes = tournamentSelection();
            newPop.add(chromosomes[0]);
        }
        population = new ArrayList<>(newPop);

    }

    private NeuralNet mutate(NeuralNet a) {
        double[] weights = a.flattenWeights();
        for (int i = 0; i < weights.length; i++) {
            if (random.nextFloat() < mutationRate) {
                weights[i] = -0.5 + (1.0) * random.nextDouble();
            }
        }
        a.unflattenWeights(weights);
        return a;
    }

    private NeuralNet[] crossover(NeuralNet parentA, NeuralNet parentB) {
        return kPoint(parentA, parentB);
    }

    private NeuralNet[] uniform(NeuralNet parentA, NeuralNet parentB) {
        double[] parentAWeights = parentA.flattenWeights();
        double[] parentBWeights = parentB.flattenWeights();
        int length = parentAWeights.length;
        double[] childAWeights = new double[length];
        double[] childBWeights = new double[length];

        boolean[] bitMask = generateBitMask(length);
        for (int i = 0; i < length; i++) {
            childAWeights[i] = bitMask[i] ? parentAWeights[i] : parentBWeights[i];
        }
        bitMask = generateBitMask(length);
        for (int i = 0; i < length; i++) {
            childBWeights[i] = bitMask[i] ? parentAWeights[i] : parentBWeights[i];
        }

        parentA.unflattenWeights(childAWeights);
        parentB.unflattenWeights(childBWeights);
        return new NeuralNet[]{parentA, parentB};
    }

    private boolean[] generateBitMask(int length) {
        boolean[] bitMask = new boolean[length];
        for (int i = 0; i < length; i++) {
            bitMask[i] = random.nextBoolean();
        }
        return bitMask;
    }

    private List<Integer> generateKPoints(int length, int max) {
        List<Integer> kPoints = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            int k = random.nextInt();
            if (!kPoints.contains(k)) {
                kPoints.add(random.nextInt(max));

            }
        }
        return kPoints;
    }

    private NeuralNet[] kPoint(NeuralNet parentA, NeuralNet parentB) {
        double[] parentAWeights = parentA.flattenWeights();
        double[] parentBWeights = parentB.flattenWeights();
        int length = parentAWeights.length;
        double[] childAWeights = new double[length];
        double[] childBWeights = new double[length];

        List<Integer> kPoints = generateKPoints(random.nextInt(length / 20), length);

        int prevK = 0;
        int k = 0;
        while (!kPoints.isEmpty()) {
            k = kPoints.remove(0);
            for (int i = prevK; i < k; i++) {
                childAWeights[i] = parentBWeights[i];
                childBWeights[i] = parentAWeights[i];
            }
            prevK = k;
            if (!kPoints.isEmpty()) {
                k = kPoints.remove(0);
                for (int i = prevK; i < k; i++) {
                    childAWeights[i] = parentAWeights[i];
                    childBWeights[i] = parentBWeights[i];
                }
                prevK = k;
            }
        }

        parentA.unflattenWeights(childAWeights);
        parentB.unflattenWeights(childBWeights);

        return new NeuralNet[]{parentA, parentB};
    }

    private NeuralNet[] onePoint(NeuralNet parentA, NeuralNet parentB) {
        double[] parentAWeights = parentA.flattenWeights();
        double[] parentBWeights = parentB.flattenWeights();
        int length = parentAWeights.length;
        double[] childAWeights = new double[length];
        double[] childBWeights = new double[length];

        int position = random.nextInt(length);

        for (int i = 0; i < position; i++) {
            childAWeights[i] = parentBWeights[i];
            childBWeights[i] = parentAWeights[i];
        }
        for (int i = position; i < length; i++) {
            childAWeights[i] = parentAWeights[i];
            childBWeights[i] = parentBWeights[i];
        }

        parentA.unflattenWeights(childAWeights);
        parentB.unflattenWeights(childBWeights);

        return new NeuralNet[]{parentA, parentB};
    }

    private NeuralNet[] tournamentSelection() {
        List<NeuralNet> tourneyPop = new ArrayList<>();
        for (int i = 0; i < tourneySize; i++) {
            int pos = random.nextInt(popSize - 1);
            tourneyPop.add(population.get(pos).clone());
        }
        Collections.sort(tourneyPop);
        return new NeuralNet[]{tourneyPop.get(0), tourneyPop.get(1)};
    }

    private void updateMutationRate() {
        mutationRate = 0.75 - (population.get(0).fitness - population.get(popSize - 1).fitness) * 0.75;
    }

    private void updateCrossoverRate() {
        crossoverRate = 0.25 + (population.get(0).fitness - population.get(popSize - 1).fitness) * 0.5;
    }

    public NeuralNet fittest(List<NeuralNet> population) {
        Collections.sort(population);
        updateMutationRate();
        updateCrossoverRate();
        return population.get(0);
    }

}

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
    public int numEpochs;
    private Random random;
    public List<NeuralNet> population;

    public GA(List<NeuralNet> population, double crossoverRate, double mutationRate, int popSize, int elitism, int tourneySize, int numEpochs, long seed) {
        this.crossoverRate = crossoverRate;
        this.mutationRate = mutationRate;
        this.population = population;
        this.popSize = popSize;
        this.tourneySize = tourneySize;
        random = new Random(seed);
    }

    public void train() {
        for (int i = 0; i < numEpochs; i++) {
            //evaluatePopulation();
            selectionProcess();
        }
    }
    
    public void evaluatePopulation(double[][] input, int[] classifications){
        population.forEach(individual -> {
            int index = random.nextInt(classifications.length);
            individual.propogateForwardsWithError(input[index], classifications[index]);
        });
    }
    
    public void evaluatePopulation(double[] input, int classifications){
        population.forEach(individual -> {
            individual.propogateForwardsWithError(input, classifications);
        });
    }
    
    public void selectionProcess() {
        List<NeuralNet> newPop = new ArrayList<>();
        Collections.sort(population);

        for (int i = 0; i < elitism; i++) {
            newPop.add(population.get(i));
        }

        while (newPop.size() < popSize) {

            NeuralNet chromosomeA = tournamentSelection();
            NeuralNet chromosomeB = tournamentSelection();

            if (random.nextFloat() < crossoverRate) {
                NeuralNet[] children;
                children = crossover(chromosomeA, chromosomeB);
                chromosomeA = children[0];
                chromosomeB = children[1];
            }

            if (random.nextFloat() < mutationRate) {
                chromosomeA = mutate(chromosomeA);
            }

            if (random.nextFloat() < mutationRate) {
                chromosomeB = mutate(chromosomeB);
            }
            newPop.add(chromosomeA);
            newPop.add(chromosomeB);
        }

        if (newPop.size() < popSize) {
            NeuralNet chromosome = tournamentSelection();
            newPop.add(chromosome);
        }

        population = newPop;
    }

    public NeuralNet mutate(NeuralNet a) {
        double[] weights = a.flattenWeights();
        for (int i = 0; i < weights.length; i++) {
            if (random.nextFloat() < mutationRate * mutationRate) {
                weights[i] = -0.5 + (1.0) * random.nextDouble();
            }
        }
        a.unflattenWeights(weights);
        return a;
    }

    public NeuralNet[] crossover(NeuralNet parentA, NeuralNet parentB) {
        double[] parentAWeights = parentA.flattenWeights();
        double[] parentBWeights = parentB.flattenWeights();
        double[] childAWeights = new double[parentAWeights.length];
        double[] childBWeights = new double[parentBWeights.length];

        int position = random.nextInt(parentAWeights.length);

        for (int i = 0; i < position; i++) {
            childAWeights[i] = parentBWeights[i];
            childBWeights[i] = parentAWeights[i];
        }
        for (int i = position; i < parentAWeights.length; i++) {
            childAWeights[i] = parentAWeights[i];
            childBWeights[i] = parentBWeights[i];
        }

        parentA.unflattenWeights(childAWeights);
        parentB.unflattenWeights(childBWeights);

        return new NeuralNet[]{parentA, parentB};
    }

    public NeuralNet tournamentSelection() {
        List<NeuralNet> tourneyPop = new ArrayList<>();
        for (int i = 0; i < tourneySize; i++) {
            tourneyPop.add(population.get(random.nextInt(popSize - 1)));
        }
        return fittest(tourneyPop);
    }

    public NeuralNet fittest(List<NeuralNet> population) {
        Collections.sort(population);
        return population.get(0);
    }

}

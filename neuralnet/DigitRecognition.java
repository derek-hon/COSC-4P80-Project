/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 *
 * @author Evan Delo
 */
public class DigitRecognition {

    public static void main(String args[]) throws FileNotFoundException {

        long seed = 123456789;

        List<InputInstance> trainData = new ArrayList<>();
        int numTrain = 700;
        int numTest = 400;
        for (int i = 0; i < 10; i++) {
            File f = new File("E:\\School\\Completed_Courses\\COSC_4P76\\Assign_1\\Neural_Net\\a1digits\\digit_train_" + i + ".txt");
            Scanner s = new Scanner(f);
            for (int j = 0; j < numTrain; j++) {
                String line = s.nextLine();
                String[] d = line.split(",");
                double[] temp = new double[64];
                for (int k = 0; k < 64; k++) {
                    temp[k] = Double.parseDouble(d[k]);
                }
                trainData.add(new InputInstance(temp, i));
            }
        }

        Random rand = new Random(seed);

        List<InputInstance> testData = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            File f = new File("E:\\School\\Completed_Courses\\COSC_4P76\\Assign_1\\Neural_Net\\a1digits\\digit_test_" + i + ".txt");
            Scanner s = new Scanner(f);
            for (int j = 0; j < numTest; j++) {
                String line = s.nextLine();
                String[] d = line.split(",");
                double[] temp = new double[64];
                for (int k = 0; k < 64; k++) {
                    temp[k] = Double.parseDouble(d[k]);
                }
                testData.add(new InputInstance(temp, i));
            }
        }

        int popSize = 50;

        GA ga = new GA(0.75, 0.6, popSize, 2, 4, rand);

        for (int k = 0; k < 1000; k++) {
            ga.evaluatePopulation(trainData);
            ga.selectionProcess();
            System.out.println(k + " " + ga.fittest(ga.population).fitness + " " + ga.population.get(popSize - 1).fitness + " " + ga.mutationRate);
        }

        ga.evaluatePopulation(testData);
        System.out.println(ga.fittest(ga.population).fitness + " " + ga.population.get(popSize - 1).fitness + " " + ga.mutationRate);

    }

}

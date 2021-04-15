/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

/**
 *
 * @author Evan Delo
 */
public class DigitRecognition {

    public static void main(String args[]) throws FileNotFoundException {

        int numTrain = 700;
        double[][][] trainData = new double[10][numTrain][64];
        int numTest = 400;
        double[][][] testData = new double[10][numTest][64];
        for (int i = 0; i < 10; i++) {
            File f = new File("E:\\School\\Completed_Courses\\COSC_4P76\\Assign_1\\Neural_Net\\a1digits\\digit_train_" + i + ".txt");
            Scanner s = new Scanner(f);
            for (int j = 0; j < trainData[i].length; j++) {
                String line = s.nextLine();
                String[] d = line.split(",");
                for (int k = 0; k < trainData[i][j].length; k++) {
                    trainData[i][j][k] = Double.parseDouble(d[k]);
                }
            }
        }

//        List<double[][]> intList = Arrays.asList(trainData);
//
//        Collections.shuffle(intList);
//
//        intList.toArray(trainData);
        for (int i = 0; i < 10; i++) {
            File f = new File("E:\\School\\Completed_Courses\\COSC_4P76\\Assign_1\\Neural_Net\\a1digits\\digit_test_" + i + ".txt");
            Scanner s = new Scanner(f);
            for (int j = 0; j < testData[i].length; j++) {
                String line = s.nextLine();
                String[] d = line.split(",");
                for (int k = 0; k < testData[i][j].length; k++) {
                    testData[i][j][k] = Double.parseDouble(d[k]);
                }
            }
        }

        GA ga = new GA(0.8, 0.01, 50, 5, 4, 700, 2);

        for (int j = 0; j < 700; j++) {
            for (int i = 0; i < 2; i++) {
                ga.evaluatePopulation(trainData[i][j], i);
                ga.selectionProcess();
                //System.out.println("Epoch: " + i + " " + j + " " + k);
            }
        }

        NeuralNet best = ga.fittest(ga.population);

        int count = 0;
        int total = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 400; j++) {
                int classified = best.propogateForwardsWithError(testData[i][j], i);
                System.out.println("error:" + best.error);
                if (classified == i) {
                    //System.out.println("error:" + best.error);
                    count++;
//                    System.out.println(classified + " " + i);
                }
                total++;
            }
        }
        System.out.println(count + " " + total + " " + (double) count / total);
    }

}

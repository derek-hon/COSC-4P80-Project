package neuralnet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.List;

public final class CharacterRecognition {

    public CharacterRecognition() {
        File file = new File("E:\\School\\COSC_4P80\\Project\\games.txt");
        BufferedReader reader;
        Random rand = new Random();
        try {
            reader = new BufferedReader(new FileReader(file));

            String line = "";
            ArrayList<InputInstance> inputData = new ArrayList<>();

            while ((line = reader.readLine()) != null) {
                String[] data = line.split(",");
                double[] inputValues = new double[data.length - 1];
                int character = enumerateClassification(data[0]);
                if (character != -1) {
                    for (int i = 1; i < data.length; i++) {
                        inputValues[i - 1] = Double.parseDouble(data[i]);
                    }
                    inputData.add(new InputInstance(inputValues, character));
                }

            }

            System.out.println(inputData.size());

            Collections.shuffle(inputData);
            List<InputInstance> gameTrain = inputData.subList(0, (int) (inputData.size() * .7));
            List<InputInstance> gameTest = inputData.subList((int) (inputData.size() * .7), inputData.size());

            int popSize = 50;

            GA ga = new GA(0.75, 0.6, popSize, 2, 4, rand);

            for (int k = 0; k < 1000; k++) {
//                Collections.shuffle(gameTrain, rand);
                ga.evaluatePopulation(inputData);
                ga.selectionProcess();
                System.out.println(k + " " + ga.fittest(ga.population).fitness + " " + ga.population.get(popSize - 1).fitness + " " + ga.mutationRate + " " + ga.crossoverRate);
            }

        } catch (IOException e) {
        }
    }

    int enumerateClassification(String input) {
        switch (input) {
            case "FOX":
                return 0;
            case "FALCO":
                return 1;
            case "CAPTAIN_FALCON":
                return 2;
            case "SHEIK":
                return 3;
            case "MARTH":
                return 4;
            default:
                return -1;
        }
    }

    public static void main(String[] args) {
        CharacterRecognition characterRecognition = new CharacterRecognition();
    }
}

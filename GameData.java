import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.List;

public class GameData {
    public GameData() {
        File file = new File("./games.txt");
        BufferedReader reader;
        Random rand = new Random();
        try {
            reader = new BufferedReader(new FileReader(file));

            String line = "";
            int[] layers = new int[] {384, 477, 573, 4};
            ArrayList<GameInput> inputData = new ArrayList<>();

            while ((line = reader.readLine()) != null ) {
                String[] data = line.split(",");
                double[] inputValues = new double[data.length - 1];
                for (int i = 1 ; i < data.length ; i ++) {
                    inputValues[i - 1] = Double.parseDouble(data[i]);
                }
                inputData.add(new GameInput(enumerateClassification(data[0]), inputValues));
            }
            Collections.shuffle(inputData);
            List<GameInput> gameTrain = inputData.subList(0, (int) (inputData.size() * .7));
            List<GameInput> gameTest = inputData.subList((int) (inputData.size() * .7), inputData.size());

            List<NeuralNet> GAMembers = new ArrayList<>();
            for (int i = 0 ; i < 100 ; i ++)
                GAMembers.add(new NeuralNet(layers, rand));

            GA test = new GA(GAMembers, 0.9, 0.1, 1000, 10, 4, 10, 50, gameTrain, rand, gameTest);
            test.train();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    int enumerateClassification(String input) {
        switch(input) {
            case "FOX":
                return 1;
            case "FALCO":
                return 2;
            case "CAPTAIN_FALCON":
                return 3;
            case "SHEIK":
                return 4;
            default:
                throw new Error("Unexpected value");
        }
    }
    public static void main(String[] args) {
        new GameData();
    }
}

class GameInput {
    int answer;
    double[] input;

    public GameInput(int answer, double[] input) {
        this.answer = answer;
        this.input = input;
    }

    int getAnswer() {
        return this.answer;
    }

    double[] getInput() {
        return this.input;
    }
}

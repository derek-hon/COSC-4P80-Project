/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

/**
 *
 * @author Evan Delo
 */
public class InputInstance {

    public double[] values;
    public int classification;

    public InputInstance(double[] values, int classification) {
        this.values = values;
        this.classification = classification;
    }

}

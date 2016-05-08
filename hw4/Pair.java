package hw4;

import weka.core.Instance;

// this class created to store pairs of instance and distance
public class Pair {
    public Instance instance;
    public double distance;

    public Pair(Instance instance, double distance) {
        this.instance = instance;
        this.distance = distance;
    }
}

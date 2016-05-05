package hw4;

import weka.core.Instance;

public class Pair {
    public Instance instance;
    public double distance;

    public Pair(Instance instance, double distance) {
        this.instance = instance;
        this.distance = distance;
    }
}

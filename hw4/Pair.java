package hw4;

import weka.core.Instance;

/**
 * Created by arnonnir on 5/4/16.
 */
public class Pair {
    public Instance instance;
    public double distance;

    public Pair(Instance instance, double distance) {
        this.instance = instance;
        this.distance = distance;
    }
}

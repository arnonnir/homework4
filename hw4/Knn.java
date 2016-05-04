package hw4;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map.Entry;


public class Knn extends Classifier {

    private final static int NUM_FOLDS = 10;
    private String M_MODE = "";
    Instances m_trainingInstances;

    public String getM_MODE() {
        return M_MODE;
    }

    public void setM_MODE(String m_MODE) {
        M_MODE = m_MODE;
    }


    public void buildClassifier(Instances instances) throws Exception {
        switch (M_MODE) {
            case "none":
                noEdit(instances);
                break;
            case "forward":
                editedForward(instances);
                break;
            case "backward":
                editedBackward(instances);
                break;
            default:
                noEdit(instances);
                break;
        }
    }

    // add random shuffle
    public double CrossValidationError(Instances trainingData, int numOfNeighbors, int pDistance, int func) {

        double crossValidationError = 0;

        for (int n = 0; n < NUM_FOLDS; n++) {
            Instances testingSet = trainingData.testCV(NUM_FOLDS, n);
            Instances trainingSet = trainingData.trainCV(NUM_FOLDS, n);
            double specificFoldError = calcAvgError(testingSet, trainingSet, numOfNeighbors, pDistance, func);

            crossValidationError += specificFoldError;
        }

        crossValidationError /= (double) NUM_FOLDS;

        return crossValidationError;
    }

    private double calcAvgError(Instances testingData, Instances trainingData, int numOfNeighbors, int pDistance, int func) {
        int numOfFoldInstances = testingData.numInstances();
        double totalFoldError = 0;

        for (int i = 0; i < numOfFoldInstances; i++) {
            double predictedValue = classify(trainingData, testingData.instance(i), numOfNeighbors, pDistance, func);
            totalFoldError += (predictedValue != testingData.instance(i).classValue()) ? 1 : 0;
        }

        double avgFoldError = totalFoldError / (double) numOfFoldInstances;

        return avgFoldError;
    }

    private double classify(Instances trainingSet, Instance instance, int numOfNeighbors, int pDistance, int func) {
        HashMap<Instance, Double> nearestNeighbors = findNearestNeighbors(trainingSet, instance, numOfNeighbors, pDistance);
        double classVote = (func == 1) ? getClassVoteResult(nearestNeighbors) : getWeightedClassVoteResult(nearestNeighbors);

        return classVote;
    }


    private HashMap<Instance, Double> findNearestNeighbors(Instances trainingData, Instance instanceToCheck, int numOfNeighbors, int pDistance) {
        HashMap<Instance, Double> allNeighbors = new HashMap<Instance, Double>();
        HashMap<Instance, Double> nearestNeighbors = new HashMap<Instance, Double>();
        int numOfInstances = trainingData.numInstances();

        for (int i = 0; i < numOfInstances; i++) {
            Instance currentInstance = trainingData.instance(i);
            double currentDistance = distance(currentInstance, instanceToCheck, pDistance);

            allNeighbors.put(currentInstance, currentDistance);
        }

        for (int i = 0; i < numOfNeighbors; i++) {
            Entry<Instance, Double> lowestNeighbor = getLowestDistanceNeighbor(allNeighbors);
            nearestNeighbors.put(lowestNeighbor.getKey(), lowestNeighbor.getValue());
            allNeighbors.remove(lowestNeighbor.getKey());
        }

        return nearestNeighbors;
    }

    private Entry<Instance, Double> getLowestDistanceNeighbor(HashMap<Instance, Double> nearestNeighbors) {
        Entry<Instance, Double> lowestNeighbor = nearestNeighbors.entrySet().iterator().next();

        for (Entry<Instance, Double> neighbor : nearestNeighbors.entrySet()) {

            if (neighbor.getValue() < lowestNeighbor.getValue()) {
                lowestNeighbor = neighbor;
            }
        }

        return lowestNeighbor;
    }

    private double distance(Instance instance1, Instance instance2, int pDistance) {
        double distance = 0;

        if (pDistance == 4) {
            lInfinityDistance(instance1, instance2);
        } else {
            lPDistance(instance1, instance2, pDistance);
        }

        return distance;
    }

    private double lPDistance(Instance instance1, Instance instance2, int pDistance) {
        int numOfFeatures = instance1.numAttributes() - 1;
        double sumOfSubstraction = 0;

        for (int i = 0; i < numOfFeatures; i++) {
            double absoluteSubstract = Math.abs(instance1.value(i) - instance2.value(i));
            sumOfSubstraction += Math.pow(absoluteSubstract, pDistance);
        }

        return Math.pow(sumOfSubstraction, (1.0 / (double) pDistance));
    }

    private double lInfinityDistance(Instance instance1, Instance instance2) {
        int numOfFeatures = instance1.numAttributes() - 1;
        double maxSubstruction = 0;

        for (int i = 0; i < numOfFeatures; i++) {
            double absoluteSubstract = Math.abs(instance1.value(i) - instance2.value(i));
            maxSubstruction = Math.max(absoluteSubstract, maxSubstruction);
        }

        return maxSubstruction;
    }

    private double getWeightedClassVoteResult(HashMap<Instance, Double> nearestNeighbors) {
        // TODO Auto-generated method stub
        return 0;
    }

    private double getClassVoteResult(HashMap<Instance, Double> nearestNeighbors) {
        int numOfClassValues = m_trainingInstances.numClasses();
        int[] countMajority = new int[numOfClassValues];
        double majorityClassValueIndex = 0;

        for (Entry<Instance, Double> neighbor : nearestNeighbors.entrySet()) {
            int classValue = (int)neighbor.getKey().classValue();
            countMajority[classValue]++;

            if(countMajority[classValue] > majorityClassValueIndex) {
                majorityClassValueIndex = classValue;
            }
        }

        return majorityClassValueIndex;
    }



    private void editedForward(Instances instances) {
    }

    private void editedBackward(Instances instances) {
    }

    private void noEdit(Instances instances) {
        m_trainingInstances = new Instances(instances);
    }

}

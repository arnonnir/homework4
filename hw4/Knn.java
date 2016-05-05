package hw4;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;


public class Knn extends Classifier {

    private final static int NUM_FOLDS = 10;
    private String M_MODE = "";
    Instances m_trainingInstances;
    public double m_totalAvgElapsedTime;

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

    public double CrossValidationError(Instances trainingData, int numOfNeighbors, int pDistance, int func) {
        Random random = new Random();
        trainingData.randomize(random);
        double crossValidationError = 0;
        long elapsedTImeToCalcError = 0;

        for (int n = 0; n < NUM_FOLDS; n++) {
            Instances testingSet = trainingData.testCV(NUM_FOLDS, n);
            Instances trainingSet = trainingData.trainCV(NUM_FOLDS, n);
            long start = System.nanoTime();
            double specificFoldError = calcAvgError(testingSet, trainingSet, numOfNeighbors, pDistance, func);
            long finish = System.nanoTime();
            elapsedTImeToCalcError += finish - start;
            crossValidationError += specificFoldError;
        }

        m_totalAvgElapsedTime = elapsedTImeToCalcError / (double) NUM_FOLDS;
        crossValidationError /= (double) NUM_FOLDS;

        return crossValidationError;
    }

    public double calcAvgError(Instances testingData, Instances trainingData, int numOfNeighbors, int pDistance, int func) {
        int numOfFoldInstances = testingData.numInstances();
        double totalFoldError = 0;

        for (int i = 0; i < numOfFoldInstances; i++) {
            double predictedValue = classify(trainingData, testingData.instance(i), numOfNeighbors, pDistance, func);
            totalFoldError += (predictedValue != testingData.instance(i).classValue()) ? 1 : 0;
        }

        double avgFoldError = totalFoldError / (double)numOfFoldInstances;

        return avgFoldError;
    }

    private double classify(Instances trainingSet, Instance instance, int numOfNeighbors, int pDistance, int func) {
        ArrayList<Pair> nearestNeighbors = findNearestNeighbors(trainingSet, instance, numOfNeighbors, pDistance);
        double classVote = (func == 1) ? getClassVoteResult(nearestNeighbors) : getWeightedClassVoteResult(nearestNeighbors);

        return classVote;
    }


    private ArrayList<Pair> findNearestNeighbors(Instances trainingData, Instance instanceToCheck, int numOfNeighbors, int pDistance) {
        ArrayList<Pair> allNeighbors = new ArrayList<Pair>();
        ArrayList<Pair> nearestNeighbors = new ArrayList<Pair>();
        int numOfInstances = trainingData.numInstances();

        for (int i = 0; i < numOfInstances; i++) {
            Instance currentInstance = trainingData.instance(i);
            double currentDistance = distance(currentInstance, instanceToCheck, pDistance);

            allNeighbors.add(new Pair(currentInstance, currentDistance));
        }

        for (int i = 0; i < numOfNeighbors; i++) {
            Pair lowestNeighbor = getLowestDistanceNeighbor(allNeighbors);
            nearestNeighbors.add(new Pair(lowestNeighbor.instance, lowestNeighbor.distance));
            allNeighbors.remove(lowestNeighbor);
        }

        return nearestNeighbors;
    }

    private Pair getLowestDistanceNeighbor(ArrayList<Pair> nearestNeighbors) {
        Pair lowestNeighbor = nearestNeighbors.get(0);

        for (Pair neighbor : nearestNeighbors) {

            if (neighbor.distance < lowestNeighbor.distance) {
                lowestNeighbor = neighbor;
            }
        }

        return lowestNeighbor;
    }

    private double distance(Instance instance1, Instance instance2, int pDistance) {
        double distance = 0;

        if (pDistance == 4) {
            distance = lInfinityDistance(instance1, instance2);
        } else {
            distance = lPDistance(instance1, instance2, pDistance);
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

    private double getWeightedClassVoteResult(ArrayList<Pair> nearestNeighbors) {
        return calculateClassVoteResult(nearestNeighbors, 2);
    }

    private double getClassVoteResult(ArrayList<Pair> nearestNeighbors) {
        return calculateClassVoteResult(nearestNeighbors, 1);
    }

    private double calculateClassVoteResult(ArrayList<Pair> nearestNeighbors, int func) {
        int numOfClassValues = m_trainingInstances.numClasses();
        double[] countMajority = new double[numOfClassValues];
        double majorityClassValueIndex = 0;

        for (Pair neighbor : nearestNeighbors) {
            int classValue = (int)neighbor.instance.classValue();
            countMajority[classValue] += (func == 1) ? 1 : 1.0 / Math.pow(neighbor.distance, 2);

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

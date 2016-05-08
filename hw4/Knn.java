package hw4;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;


public class Knn extends Classifier {

    private final static int NUM_FOLDS = 10;
    private String M_MODE = "";
    private configurationParams m_bestParams;
    Instances m_trainingInstances;
    private double m_avgElapsedTime;

    public String getM_MODE() {
        return M_MODE;
    }

    public void setM_MODE(String m_MODE) {
        M_MODE = m_MODE;
    }
    
    public double getAvgElapsedTime() {
    	return m_avgElapsedTime;
    }
    
    public void setConfigParams(int k, int p, int func) {
    	this.m_bestParams = new configurationParams(k, p, func);
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

    public double CrossValidationError(Instances trainingData) {
    	// shuffle instances before divide them 
        Random random = new Random();
        trainingData.randomize(random);
        double crossValidationError = 0;
        long elapsedTImeToCalcError = 0;

        int numOfInstances = trainingData.numInstances();
        // if the number of instances is lower then num of folds (10),
        // assign num of folds to be the number of instances
        int numOfFolds = (numOfInstances < NUM_FOLDS) ? numOfInstances : NUM_FOLDS;
        for (int n = 0; n < numOfFolds; n++) {
        	// these methods divide the data so that fold n is the testing data, and the rest is the training
            Instances testingSet = trainingData.testCV(numOfFolds, n);
            Instances trainingSet = trainingData.trainCV(numOfFolds, n);
            long start = System.nanoTime();
            double specificFoldError = calcAvgError(testingSet, trainingSet);
            long finish = System.nanoTime();
            elapsedTImeToCalcError += finish - start;
            crossValidationError += specificFoldError;
        }

        m_avgElapsedTime = elapsedTImeToCalcError / (double) numOfFolds;
        crossValidationError /= (double) numOfFolds;

        return crossValidationError;
    }

    public double calcAvgError(Instances testingData, Instances trainingData) {
        int numOfFoldInstances = testingData.numInstances();
        double totalFoldError = 0;

        for (int i = 0; i < numOfFoldInstances; i++) { // run throw all instances in the specific fold
        	double classValue = testingData.instance(i).classValue();
            double predictedValue = classify(trainingData, testingData.instance(i));
            totalFoldError += (predictedValue != classValue) ? 1 : 0;
        }

        double avgFoldError = totalFoldError / (double)numOfFoldInstances;

        return avgFoldError;
    }

    private double classify(Instances trainingSet, Instance instance) {
    	// make list if pairs (neighbor, distance) of the k nearest
        ArrayList<Pair> nearestNeighbors = findNearestNeighbors(trainingSet, instance);
        // if function 1 is the best, use non weigthed vote. otherwise, use weighted
        double classVote = (m_bestParams.function == 1) ? getClassVoteResult(nearestNeighbors) : getWeightedClassVoteResult(nearestNeighbors);

        return classVote;
    }

    private ArrayList<Pair> findNearestNeighbors(Instances trainingData, Instance instanceToCheck) {
        ArrayList<Pair> allNeighbors = new ArrayList<Pair>(); // make list of all instances
        ArrayList<Pair> nearestNeighbors = new ArrayList<Pair>(); // make list of only the k nearest within all instances
        int numOfInstances = trainingData.numInstances();

        for (int i = 0; i < numOfInstances; i++) {
            Instance currentInstance = trainingData.instance(i);
            // for each instance x, check the distance between x and instance to
            // check and add him to the all instances list
            double currentDistance = distance(currentInstance, instanceToCheck);
            allNeighbors.add(new Pair(currentInstance, currentDistance));
        }

        for (int i = 0; i < this.m_bestParams.k; i++) { // run k times
            Pair lowestNeighbor = getLowestDistanceNeighbor(allNeighbors); // each time take the nearest neighbor
            nearestNeighbors.add(new Pair(lowestNeighbor.instance, lowestNeighbor.distance)); // add him to k nearest
            allNeighbors.remove(lowestNeighbor); // remove him from the list, to get the i't nearest in the next iteration
        }

        return nearestNeighbors;
    }
    
    // this method returns Pair of the neighbor with the lowest distance
    private Pair getLowestDistanceNeighbor(ArrayList<Pair> nearestNeighbors) { 
        Pair lowestNeighbor = nearestNeighbors.get(0);
        for (Pair neighbor : nearestNeighbors) {
            if (neighbor.distance < lowestNeighbor.distance) {
                lowestNeighbor = neighbor;
            }
        }

        return lowestNeighbor;
    }

    private double distance(Instance instance1, Instance instance2) {
        double distance = 0;
        // we assign p=4 to be l-infinity
        if (this.m_bestParams.p == 4) {
            distance = lInfinityDistance(instance1, instance2);
        // else p=1,2,3
        } else {
            distance = lPDistance(instance1, instance2, this.m_bestParams.p);
        }

        return distance;
    }

    private double lPDistance(Instance instance1, Instance instance2, int pDistance) {
        int numOfFeatures = instance1.numAttributes() - 1;
        double sumOfSubstraction = 0;

        for (int i = 0; i < numOfFeatures; i++) { // calculate distance according to the formula
            double absoluteSubstract = Math.abs(instance1.value(i) - instance2.value(i));
            sumOfSubstraction += Math.pow(absoluteSubstract, pDistance);
        }

        return Math.pow(sumOfSubstraction, (1.0 / (double) pDistance));
    }

    private double lInfinityDistance(Instance instance1, Instance instance2) {
        int numOfFeatures = instance1.numAttributes() - 1;
        double maxSubstruction = 0;

        for (int i = 0; i < numOfFeatures; i++) { // calculate distance according to the formula
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
        // create array to count how much instances for each class value
        double[] countMajority = new double[numOfClassValues];
        double majorityClassValueIndex = 0;

        for (Pair neighbor : nearestNeighbors) {
            int classValue = (int)neighbor.instance.classValue();
            // func = 1 refers to non-weigthed, func = 2 refer to weighted
            countMajority[classValue] += (func == 1) ? 1 : 1.0 / Math.pow(neighbor.distance, 2);
            
            // store the index with the major 
            if(countMajority[classValue] > majorityClassValueIndex) {
                majorityClassValueIndex = classValue;
            }
        }

        return majorityClassValueIndex;
    }

    private void editedForward(Instances instances) {
    	int numOfInstances = instances.numInstances();
    	// assign training set to be empty set
    	m_trainingInstances = new Instances(instances, numOfInstances);

		for (int i = 0; i < numOfInstances; i++) { // run throw all original instances
			Instance currentInstance = instances.instance(i);
			double classValue = currentInstance.classValue();

			// add the first k instances
			if (i < this.m_bestParams.k) { 
				m_trainingInstances.add(currentInstance);
			} else {
				double predictedValue = classify(m_trainingInstances, currentInstance);
				// after first k, add instance that isn't correctly classified
				if (classValue != predictedValue) {
					m_trainingInstances.add(currentInstance);
				}
			}
		}
    }

    private void editedBackward(Instances instances) {
    	int numOfInstances = instances.numInstances();
    	// assign training set to be all instances
    	m_trainingInstances = new Instances(instances);

		for (int i = 0; i < numOfInstances; i++) { // run throw all
			Instance currentInstance = instances.instance(i);
			// create list of all instances exclude the instance we check
			Instances instancesExcludeCurrent = new Instances(m_trainingInstances);
			instancesExcludeCurrent.delete(i - (numOfInstances - m_trainingInstances.numInstances()));
			
			double classValue = currentInstance.classValue();
			double predictedValue = classify(instancesExcludeCurrent, currentInstance);
			if (classValue == predictedValue) {
				// delete from training data the instance in case that he classified currectly
				m_trainingInstances.delete(i - (numOfInstances - m_trainingInstances.numInstances()));;
			}
		}
    }

    private void noEdit(Instances instances) {
        m_trainingInstances = new Instances(instances);
    }

}

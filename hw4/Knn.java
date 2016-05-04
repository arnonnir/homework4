package hw4;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;


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

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		switch (M_MODE){
		case "none":
			noEdit(arg0);
			break;
		case "forward":
			editedForward(arg0);
			break;
		case "backward":
			editedBackward(arg0);
			break;
		default:
			noEdit(arg0);
			break;
		}
	}
	
	public double CrossValidationError(Instances trainingData, int numOfNeighbors, int pDistance, int func) {
		
		double crossValidationError = 0;
		
		for (int n = 0; n < NUM_FOLDS; n++) {
			   Instances trainingSet = trainingData.trainCV(NUM_FOLDS, n);
			   Instances testingSet = trainingData.testCV(NUM_FOLDS, n);
			   double specificFoldError = 0;
			   
			   int numOfFoldInstances = testingSet.numInstances();
			   for (int i = 0; i < numOfFoldInstances; i++) {
				   double[] nearestNeighbors = findNearestNeighbors(trainingSet, trainingData.instance(i), numOfNeighbors, pDistance);		
				   double classVote = (func == 1) ? getClassVoteResult(nearestNeighbors) : getWeightedClassVoteResult(nearestNeighbors);
				   specificFoldError += (classVote != testingSet.instance(i).classValue()) ? 1 : 0;
			   }
			   
			   specificFoldError /= (double)numOfFoldInstances;
			   crossValidationError += specificFoldError;
		}
					   
		return crossValidationError /= (double)NUM_FOLDS;
	}
	
	private double getWeightedClassVoteResult(double[] nearestNeighbors) {
		// TODO Auto-generated method stub
		return 0;
	}

	private double getClassVoteResult(double[] nearestNeighbors) {
		// TODO Auto-generated method stub
		return 0;
	}

	private double[] findNearestNeighbors(Instances allNeighbors, Instance instanceToCheck, int numOfNeighbors, int pDistance) {
		// TODO Auto-generated method stub
		return null;
	}

	private void editedForward(Instances instances) {
	}
	
	private void editedBackward(Instances instances) {
	}
	
	private void noEdit(Instances instances) {
		m_trainingInstances = new Instances(instances);
	}

}

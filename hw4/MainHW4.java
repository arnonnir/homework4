package hw4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW4 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
	
	public static void main (String [] args) throws Exception{
		String glassFile = "glass";
		String cancerFile = "cancer";

		configurationParams configParams = run(glassFile);
		run(cancerFile);

		Knn nonEditedKnn = initialize("glass", "none");
		double nonEditedCrossValidationError = nonEditedKnn.CrossValidationError(nonEditedKnn.m_trainingInstances, configParams.k, configParams.p, configParams.function);
		double avgNonEditedElapsedTime = nonEditedKnn.m_totalAvgElapsedTime;
		Knn forwardEditedKnn = initialize("glass", "forward");
		double forwardEditedCrossValidationError = nonEditedKnn.CrossValidationError(forwardEditedKnn.m_trainingInstances, configParams.k, configParams.p, configParams.function);
		double avgForwardElapsedTime = forwardEditedKnn.m_totalAvgElapsedTime;
		Knn backwardEditedKnn = initialize("glass", "backward");
		double backwardEditedCrossValidationError = nonEditedKnn.CrossValidationError(backwardEditedKnn.m_trainingInstances, configParams.k, configParams.p, configParams.function);
		double avgBackwardElapsedTime = backwardEditedKnn.m_totalAvgElapsedTime;

		System.out.println("Cross validation error of non-edited knn on glass dataset is " + nonEditedCrossValidationError + "and the average elapsed time is " + avgNonEditedElapsedTime);
		System.out.println("Cross validation error of forwards-edited knn on glass dataset is " + forwardEditedCrossValidationError + "and the average elapsed time is " + avgForwardElapsedTime);
		System.out.println("Cross validation error of backwards-edited knn on glass dataset is " + backwardEditedCrossValidationError + "and the average elapsed time is " + avgBackwardElapsedTime);
	}

	private static Knn initialize(String fileName, String mode) throws Exception{
		BufferedReader readTraining = readDataFile("src/hw4/" + fileName + ".txt");
		Instances instancesTraining = new Instances(readTraining);
		instancesTraining.setClassIndex(instancesTraining.numAttributes() - 1);

		Knn kNearestNeighbor = new Knn();
		kNearestNeighbor.setM_MODE(mode);
		kNearestNeighbor.buildClassifier(instancesTraining);

		return kNearestNeighbor;
	}

	private static configurationParams run(String fileName) throws Exception{
		Knn kNearestNeighbor = initialize(fileName, "none");
		int k_MaxValue = 1;
		int p_MaxValue = 1;
		int func_MaxValue = 1;
		double minError = Double.MAX_VALUE;

		for (int k = 1; k < 31; k++) {
			for (int p = 1; p < 5; p++) {
				for (int func = 1; func < 3; func++) {
					double currentParamsError = kNearestNeighbor.CrossValidationError(kNearestNeighbor.m_trainingInstances, k, p, func);

					if (currentParamsError < minError) {
						k_MaxValue = k;
						p_MaxValue = p;
						func_MaxValue = func;
						minError = currentParamsError;
					}
				}
			}
		}

		System.out.println("Cross validation error with K = " + k_MaxValue + ", p = " + p_MaxValue + ", vote function = " + func_MaxValue + " for " + fileName +" data is: " + minError);

		return new configurationParams(k_MaxValue, p_MaxValue, func_MaxValue);
	}

}

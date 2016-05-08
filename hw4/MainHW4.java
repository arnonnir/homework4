package hw4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

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

		// find best params for glass.txt
		configurationParams bestParamsForGlass = run(glassFile);
		
		// find best params for cancer.txt
		configurationParams bestParamsForCancer = run(cancerFile);

		Instances instancesTraining = initialize("glass");
		Knn kNearestNeighbor = new Knn();
		// set the best params we find earlier
		kNearestNeighbor.setConfigParams(bestParamsForGlass.k, bestParamsForGlass.p, bestParamsForGlass.function);
		
		// run 3 times, for all modes
		for (int i = 0; i < 3; i++) {
			if (i == 0) {
				kNearestNeighbor.setM_MODE("none");
			} else if (i == 1) {
				kNearestNeighbor.setM_MODE("forward");
			} else if (i == 2) {
				kNearestNeighbor.setM_MODE("backward");
			}
			
			kNearestNeighbor.buildClassifier(instancesTraining);
			double currentModeCrossValidationError = kNearestNeighbor.CrossValidationError(kNearestNeighbor.m_trainingInstances);
			double currentModeAvgElapsedTime = kNearestNeighbor.getAvgElapsedTime();
			System.out.println("Cross validation error of " + kNearestNeighbor.getM_MODE() + " edited knn on glass dataset is " + currentModeCrossValidationError + " and the average elapsed time is " + currentModeAvgElapsedTime);
		}
	}

	private static Instances initialize(String fileName) throws Exception{
		BufferedReader readTraining = readDataFile("src/hw4/" + fileName + ".txt");
		Instances instancesTraining = new Instances(readTraining);
		instancesTraining.setClassIndex(instancesTraining.numAttributes() - 1);
		
		if (fileName.equals("glass")) {
			instancesTraining.deleteAttributeAt(0);
		}
		
		return instancesTraining;
	}

	// this method runs with all combinations of params and calculate the cross validation for each.
	// returns the best configuration of params
	private static configurationParams run(String fileName) throws Exception{
		Instances instancesTraining = initialize(fileName);
		Knn kNearestNeighbor = new Knn();
		kNearestNeighbor.setM_MODE("none");
		kNearestNeighbor.buildClassifier(instancesTraining);
		
		int bestK = 1;
		int bestP = 1;
		int bestFunc = 1;
		double minError = Double.MAX_VALUE;

		for (int k = 1; k < 31; k++) {
			for (int p = 1; p < 5; p++) {
				for (int func = 1; func < 3; func++) {
					kNearestNeighbor.setConfigParams(k, p, func);
					double currentParamsError = kNearestNeighbor.CrossValidationError(kNearestNeighbor.m_trainingInstances);

					if (currentParamsError < minError) {
						bestK = k;
						bestP = p;
						bestFunc = func;
						minError = currentParamsError;
					}
				}
			}
		}

		String bestFuncName = (bestFunc == 1) ? "Non-Weighted" : "Weighted";
		System.out.println("Cross validation error with K = " + bestK + ", p = " + ((bestP == 4) ? "infinity" : bestP) + ", vote function = " + bestFuncName + " for " + fileName +" data is: " + minError);

		return new configurationParams(bestK, bestP, bestFunc);
	}

}

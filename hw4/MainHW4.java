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
		System.out.println(1/3);
		BufferedReader readTraining = readDataFile("/Users/arnonnir/IdeaProjects/HomeWork4/src/hw4/glass.txt");
		Instances instancesTraining = new Instances(readTraining);
		instancesTraining.setClassIndex(instancesTraining.numAttributes() - 1);

		Knn kNearestNeighbor = new Knn();
		kNearestNeighbor.setM_MODE("");
		kNearestNeighbor.buildClassifier(instancesTraining);

		choseParameters(kNearestNeighbor, instancesTraining);
		
	
		
		
	}

	private static void choseParameters(Knn kNearestNeighbor, Instances trainingData) {
		int k_MaxValue = 1;
		int p_MaxValue = 1;
		int func_MaxValue = 1;
		double minError = Integer.MAX_VALUE;
		
		for (int k = 1; k < 31; k++) {
			   for (int p = 1; p < 4; p++) {
				   for (int func = 1; func < 3; func++) {
					   double currentParamsError = kNearestNeighbor.CrossValidationError(trainingData, k, p, func);

					   if (currentParamsError < minError) {
						   k_MaxValue = k;
						   p_MaxValue = p;
						   func_MaxValue = func;
						   minError = currentParamsError;
					   }
				   }   
			   }
		}
		
	}

}

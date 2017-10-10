import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class marginPerceptron {

	static ArrayList<Double[]> fourFoldTrainData = new ArrayList<>();
	static ArrayList<Double[]> fourFoldTestData = new ArrayList<>();
	static ArrayList<Double[]> fullTrainData = new ArrayList<>();
	static ArrayList<Double[]> devData = new ArrayList<>();
	static ArrayList<Double[]> testData = new ArrayList<>();
	static int DIMENSIONALITY = 69; 
	static DecimalFormat df = new DecimalFormat("###.0");
	
	public static void main(String[] args) {
		String splitTrainFilePath = args[0];
		String fullTrainFilePath = args[1];
		String fullDevFilePath = args[2];
		String fullTestFilePath = args[3];
		
		double[] bestValues = runSimplePerceptronCrossValidTest(splitTrainFilePath);
		buildTrainData(fullTrainFilePath);
		buildDevData(fullDevFilePath);
		buildTestData(fullTestFilePath);
		
		simplePerceptronFullDevAndTest(bestValues[0], bestValues[1]);
	}

	private static double[] runSimplePerceptronCrossValidTest(String splitTrainFilePath) {
		Double[][] validationResults = new Double[9][8];
		validationResults[0][0] = 1.0;
		validationResults[3][0] = 0.1;
		validationResults[6][0] = 0.01;
		
		makeFoldTrainDataFromFile(splitTrainFilePath, 0,1,2,3,4);
		runPerceptronCrossValidAccuracy(4, validationResults);
		
		makeFoldTrainDataFromFile(splitTrainFilePath, 0,1,2,4,3);
		runPerceptronCrossValidAccuracy(3, validationResults);
		
		makeFoldTrainDataFromFile(splitTrainFilePath, 0,1,3,4,2);
		runPerceptronCrossValidAccuracy(2, validationResults);
		
		makeFoldTrainDataFromFile(splitTrainFilePath, 0,2,3,4,1);
		runPerceptronCrossValidAccuracy(1, validationResults);
		
		makeFoldTrainDataFromFile(splitTrainFilePath, 1,2,3,4,0);
		runPerceptronCrossValidAccuracy(0, validationResults);
		
		// Compute average for all Learn rates
		for(int i = 0; i < 9; i++) {
			double avg = 0.0;
			for(int j = 2; j < 7; j++) {
				avg += validationResults[i][j];
			}
			validationResults[i][7] = avg/5;
		}
		
		System.out.println(String.format("%-7s %-7s %-7s %-7s %-7s %-7s %-7s %-7s", "L. Rate", "margin",  
				"% 00 ", "% 01 ", "% 02 ", "% 03 ", "% 04 ", "% avg "));
		System.out.println("---------------------------------------------------------");
		
		for(int i = 0; i < 9; i++) {
			System.out.print(String.format("%-7s ", validationResults[i][0]));
			System.out.print(String.format("%-7s ", validationResults[i][1]));
			for(int j = 2; j < 8; j++) {
				System.out.print(String.format("%-7s ", df.format(validationResults[i][j])));
			}
			System.out.println();
		}
		
		Double[] avgLR = new Double[3];
		avgLR[0] = (validationResults[0][7] + validationResults[1][7] + validationResults[2][7])/3.0;
		avgLR[1] = (validationResults[3][7] + validationResults[4][7] + validationResults[5][7])/3.0;
		avgLR[2] = (validationResults[6][7] + validationResults[7][7] + validationResults[8][7])/3.0;
		
		Double[] avgMargin = new Double[3];
		avgMargin[0] = (validationResults[0][7] + validationResults[3][7] + validationResults[6][7])/3.0;
		avgMargin[1] = (validationResults[1][7] + validationResults[4][7] + validationResults[7][7])/3.0;
		avgMargin[2] = (validationResults[2][7] + validationResults[5][7] + validationResults[8][7])/3.0;
		
		// Find best LR
		double bestLR = 0.1;
		if(avgLR[0] > avgLR[1]) {
			bestLR = 1.0;
			if(avgLR[0] > avgLR[2]) {
				bestLR = 1.0;
				System.out.println("\nCross-valid accuracy for best l-rate(" + bestLR + ") = " 
						+ df.format(avgLR[0]));
			}
			else {
				bestLR = 0.01;
				System.out.println("\nCross-valid accuracy for best l-rate(" + bestLR + ") = " 
						+ df.format(avgLR[2]));
			}
		}
		else if(avgLR[1] > avgLR[2]) {
			bestLR = 0.1;
			System.out.println("\nCross-valid accuracy for best l-rate(" + bestLR + ") = " 
					+ df.format(avgLR[1]));
		}
		else {
			bestLR = 0.01;
			System.out.println("\nCross-valid accuracy for best l-rate(" + bestLR + ") = " 
					+ df.format(avgLR[2]));
		}
		
		
		
		// Find best margin
		double bestMargin = 0.1;
		if(avgMargin[0] > avgMargin[1]) {
			bestMargin = 1.0;
			if(avgMargin[0] > avgMargin[2]) {
				bestMargin = 1.0;
				System.out.println("\nCross-valid accuracy for best margin(" + bestMargin + ") = " 
						+ df.format(avgMargin[0]));
			}
			else {
				bestMargin = 0.01;
				System.out.println("\nCross-valid accuracy for best margin(" + bestMargin + ") = " 
						+ df.format(avgMargin[2]));
			}
		}
		else if(avgMargin[1] > avgMargin[2]) {
			bestMargin = 0.1;
			System.out.println("\nCross-valid accuracy for best margin(" + bestMargin + ") = " 
					+ df.format(avgMargin[1]));
		}
		else {
			bestMargin = 0.01;
			System.out.println("\nCross-valid accuracy for best margin(" + bestMargin + ") = " 
					+ df.format(avgMargin[2]));
		}
		
		System.out.println("\nBest Learning Rate for Margin Perceptron: " + bestLR);
		System.out.println("Best Margin for Margin Perceptron       : " + bestMargin);
		return new double[]{bestLR, bestMargin};
	}
	
	private static void runPerceptronCrossValidAccuracy(int splitPart, Double[][] validRes) {
		double[] learningRates = {1.0, 0.1, 0.01};
		double[] margins = {1.0, 0.1, 0.01};
		
		for(int hyperIter = 0; hyperIter < 9; hyperIter += 3) {
			for(int mIter = 0; mIter < margins.length; mIter++) {	
				validRes[hyperIter+mIter][0] = learningRates[hyperIter/3];
				validRes[hyperIter+mIter][1] = margins[mIter];
				Vector wVector = new Vector(simplePerceptron4Fold(learningRates[hyperIter/3], margins[mIter]));

				int numOfMistakes = 0;
				for(Double[] entry : fourFoldTestData) {
					double expectedY = entry[0];
					Vector xVector = new Vector(Arrays.copyOfRange(entry, 1, entry.length));

					Double t = wVector.dot(xVector);

					if(!isPredictionTrue(t, expectedY)) {
						numOfMistakes++;
					}
				}
				double accuracy = ((fourFoldTestData.size() - (double)numOfMistakes)/fourFoldTestData.size())* 100.0;
				validRes[hyperIter+mIter][splitPart+2] = accuracy;
			}
		}
	}
	
	private static boolean isPredictionTrue(Double predicted, Double expected) {
		
		if( (predicted >= 0.0 && expected < 0.0) || (predicted < 0.0 && expected >= 0.0)) {
			return false;
		}
		else {
			return true;
		}
	}

	/**
	 * Returns vector w using the passed learning rate
	 * @param lRate
	 * @return Weights vector w
	 */
	private static Double[] simplePerceptron4Fold(double lRate, double margin) {
		Double[] w = getRandomW();
		Vector wVector = new Vector(w);
		double bias = 0.01;
		
		// Train on 4-fold
		for(int i = 0; i < 10; i++) {
			Collections.shuffle(fourFoldTrainData);
			for(Double[] entry : fourFoldTrainData) {
				Double[] ex = Arrays.copyOfRange(entry, 1, entry.length);
				double y = entry[0];
				Vector xVector = new Vector(ex);
				
				if( ( (wVector.dot(xVector) + bias) * y ) < margin) {
					wVector = wVector.plus(xVector.scale(lRate*y));
					bias = bias + (lRate * y);
				}
				lRate = lRate / (1 + i);
			}
		}
		
		return wVector.getDoubleArray();
	}
	
	/**
	 * Returns vector w using the passed learning rate
	 * @param lRate
	 * @return Weights vector w
	 */
	private static Double[] simplePerceptronFullDevAndTest(double lRate, double margin) {
		Double[] w = getRandomW();
		Vector wVector = new Vector(w);
		double bias = 0.01;
		Double[] devAccuracyTable = new Double[20];	// 20 epoch
		double originalLRate = lRate;
		
		// Train on full
		for(int i = 0; i < 20; i++) {
			Collections.shuffle(fullTrainData);
			for(Double[] entry : fullTrainData) {
				Double[] ex = Arrays.copyOfRange(entry, 1, entry.length);
				double y = entry[0];
				Vector xVector = new Vector(ex);
				
				if( ( (wVector.dot(xVector) + bias) * y ) < margin) {
					wVector = wVector.plus(xVector.scale(lRate*y));
					bias = bias + (lRate * y);
				}
				lRate = lRate / (1 + i);
			}
			
			// Measure Dev accuracy
			int numOfMistakes = 0;
			for(Double[] entry : devData) {
				double expectedY = entry[0];
				Vector xVector = new Vector(Arrays.copyOfRange(entry, 1, entry.length));
				
				Double t = wVector.dot(xVector);
				
				if(!isPredictionTrue(t, expectedY)) {
					numOfMistakes++;
				}
			}
			devAccuracyTable[i] = ((devData.size() - (double)numOfMistakes)/devData.size())* 100.0;
		}
		System.out.println("\n-------------------------- DEV ACCURACY --------------------------------");
		System.out.println(String.format("%-7s %-7s %-7s %-7s %-7s %-7s %-7s %-7s", "Epoch", "% acc", "Epoch", "% acc", 
				"Epoch", "% acc", "Epoch", "% acc"));
		System.out.println("------------------------------------------------------------------------");
		for(int j = 0; j < devAccuracyTable.length/4; j++) {
			System.out.println(String.format("%-7s %-7s %-7s %-7s %-7s %-7s %-7s %-7s", j+1, df.format(devAccuracyTable[j]), 
					j+1+5, df.format(devAccuracyTable[j+5]), j+1+10, df.format(devAccuracyTable[j+10]), 
					j+1+15, df.format(devAccuracyTable[j+15])));
		}
		
		double bestDevAccuracy = 0.0;
		int bestDevEpoch = 0;
		for(int j = 0; j < devAccuracyTable.length; j++) {
			if(devAccuracyTable[j] > bestDevAccuracy) {
				bestDevAccuracy = devAccuracyTable[j];
				bestDevEpoch = j+1;
			}
		}
		
		System.out.println("\nHighest Dev Epoch: " + bestDevEpoch);
		
		wVector = new Vector(getRandomW());
		bias = 0.01;
		lRate = originalLRate;
		int updates = 0;
		
		// Train on full for best epoch and then test on test.data
		for(int i = 0; i < bestDevEpoch; i++) {
			Collections.shuffle(fullTrainData);
			for(Double[] entry : fullTrainData) {
				Double[] ex = Arrays.copyOfRange(entry, 1, entry.length);
				double y = entry[0];
				Vector xVector = new Vector(ex);

				if( ( (wVector.dot(xVector) + bias) * y ) < margin) {
					wVector = wVector.plus(xVector.scale(lRate*y));
					bias = bias + (lRate * y);
					updates++;
				}
			}
			lRate = lRate / (1 + i);
		}
		// Measure test accuracy
		int numOfMistakes = 0;
		for(Double[] entry : testData) {
			double expectedY = entry[0];
			Vector xVector = new Vector(Arrays.copyOfRange(entry, 1, entry.length));

			Double t = wVector.dot(xVector);

			if(!isPredictionTrue(t, expectedY)) {
				numOfMistakes++;
			}
		}
		double testAccuracy = ((testData.size() - (double)numOfMistakes)/testData.size())* 100.0;
		System.out.println("Best dev accuracy = " + df.format(bestDevAccuracy));
		System.out.println("Test accuracy     = " + df.format(testAccuracy));
		System.out.println("# updates for cumulative epoch = " + updates);
		System.out.println("# updates per epoch = " + updates/bestDevEpoch);
		
		return wVector.getDoubleArray();
	}
	
	private static Double[] getRandomW() {
		Double[] w = new Double[DIMENSIONALITY - 1];
		
		for(int i = 0; i < 68; i++)
			if(i % 2 == 0) {
				w[i] = 0.01;
			}
			else {
				w[i] = 0.01;
			}
		return w;
	}

	private static void buildTrainData(String fullDataFilePath) {
		int index = 0;
		try {
			BufferedReader lineReader = new BufferedReader(new FileReader(fullDataFilePath));
			String line = null;
			
			int count = 0;
			while ((line = lineReader.readLine()) != null) {
				
				String[] row = line.toLowerCase().split("\\s+");
				fullTrainData.add(new Double[DIMENSIONALITY]);
				Arrays.fill(fullTrainData.get(index), 0.0);
				
				for(int j = 0; j < row.length; j++) {
					if(j == 0) {
						fullTrainData.get(index)[0] = Double.parseDouble(row[0]);
					}
					else {
						String[] entry = row[j].split(":");
						int pos = Integer.parseInt(entry[0]);
						double value = Double.parseDouble(entry[1]);
						fullTrainData.get(index)[pos] = value;
					}
				}	
				count++;
				index++;
			}
//			System.out.println("Rows in Train" + "  : " + count);
			
			lineReader.close();
		} catch (IOException ex) {
			System.err.println(ex);
		}
	}
	
	private static void buildDevData(String fullDataFilePath) {
		int index = 0;
		try {
			BufferedReader lineReader = new BufferedReader(new FileReader(fullDataFilePath));
			String line = null;
			
			int count = 0;
			while ((line = lineReader.readLine()) != null) {
				
				String[] row = line.toLowerCase().split("\\s+");
				devData.add(new Double[DIMENSIONALITY]);
				Arrays.fill(devData.get(index), 0.0);
				
				for(int j = 0; j < row.length; j++) {
					if(j == 0) {
						devData.get(index)[0] = Double.parseDouble(row[0]);
					}
					else {
						String[] entry = row[j].split(":");
						int pos = Integer.parseInt(entry[0]);
						double value = Double.parseDouble(entry[1]);
						devData.get(index)[pos] = value;
					}
				}	
				count++;
				index++;
			}
//			System.out.println("Rows in Dev" + "  : " + count);
			
			lineReader.close();
		} catch (IOException ex) {
			System.err.println(ex);
		}
	}
	
	private static void buildTestData(String fullDataFilePath) {
		int index = 0;
		try {
			BufferedReader lineReader = new BufferedReader(new FileReader(fullDataFilePath));
			String line = null;
			
			int count = 0;
			while ((line = lineReader.readLine()) != null) {
				
				String[] row = line.toLowerCase().split("\\s+");
				testData.add(new Double[DIMENSIONALITY]);
				Arrays.fill(testData.get(index), 0.0);
				
				for(int j = 0; j < row.length; j++) {
					if(j == 0) {
						testData.get(index)[0] = Double.parseDouble(row[0]);
					}
					else {
						String[] entry = row[j].split(":");
						int pos = Integer.parseInt(entry[0]);
						double value = Double.parseDouble(entry[1]);
						testData.get(index)[pos] = value;
					}
				}	
				count++;
				index++;
			}
//			System.out.println("Rows in Test" + "  : " + count);
			
			lineReader.close();
		} catch (IOException ex) {
			System.err.println(ex);
		}
	}
	
	private static void makeFoldTrainDataFromFile(String splitTrainFilePath, 
			int p0, int p1, int p2, int p3, int holdOffPart) {
		
		String part0Path = splitTrainFilePath + "/training0" + p0 + ".data";
		String part1Path = splitTrainFilePath + "/training0" + p1 + ".data";
		String part2Path = splitTrainFilePath + "/training0" + p2 + ".data";
		String part3Path = splitTrainFilePath + "/training0" + p3 + ".data";
		String holdOffPartPath = splitTrainFilePath + "/training0" + holdOffPart + ".data";
		
		fourFoldTrainData = new ArrayList<>();
		fourFoldTestData = new ArrayList<>();
		
		int index = 0;
		for(int i = 0; i <= 3; i++) {
			String filePath = "";
			switch(i) {
			case 0:
				filePath = part0Path;
				break;
			case 1:
				filePath = part1Path;
				break;
			case 2:
				filePath = part2Path;
				break;
			case 3:
				filePath = part3Path;
				break;
			}
			
			try {
				BufferedReader lineReader = new BufferedReader(new FileReader(filePath));
				String line = null;
				
				int count = 0;
				while ((line = lineReader.readLine()) != null) {
					
					String[] row = line.toLowerCase().split("\\s+");
					fourFoldTrainData.add(new Double[DIMENSIONALITY]);
					Arrays.fill(fourFoldTrainData.get(index), 0.0);
					
					for(int j = 0; j < row.length; j++) {
						if(j == 0) {
							fourFoldTrainData.get(index)[0] = Double.parseDouble(row[0]);
						}
						else {
							String[] entry = row[j].split(":");
							int pos = Integer.parseInt(entry[0]);
							double value = Double.parseDouble(entry[1]);
							fourFoldTrainData.get(index)[pos] = value;
						}
					}	
					count++;
					index++;
				}
//				System.out.println("Rows in File " + i + ": " + count);
				
				lineReader.close();
			} catch (IOException ex) {
				System.err.println(ex);
			}
		}	
		index = 0;
		try {
			BufferedReader lineReader = new BufferedReader(new FileReader(holdOffPartPath));
			String line = null;
			
			int count = 0;
			while ((line = lineReader.readLine()) != null) {
				
				String[] row = line.toLowerCase().split("\\s+");
				fourFoldTestData.add(new Double[DIMENSIONALITY]);
				Arrays.fill(fourFoldTestData.get(index), 0.0);
				
				for(int j = 0; j < row.length; j++) {
					if(j == 0) {
						fourFoldTestData.get(index)[0] = Double.parseDouble(row[0]);
					}
					else {
						String[] entry = row[j].split(":");
						int pos = Integer.parseInt(entry[0]);
						double value = Double.parseDouble(entry[1]);
						fourFoldTestData.get(index)[pos] = value;
					}
				}	
				count++;
				index++;
			}
//			System.out.println("Rows in Hold" + "  : " + count);
			
			lineReader.close();
		} catch (IOException ex) {
			System.err.println(ex);
		}
//		System.out.println("Size of Train 4-part: " + fourFoldTrainData.size());
//		System.out.println("Size of Test 4-part : " + fourFoldTestData.size());
	}

}
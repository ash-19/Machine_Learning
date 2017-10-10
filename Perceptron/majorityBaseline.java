import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;


public class majorityBaseline {

	static ArrayList<Double[]> fourFoldTrainData = new ArrayList<>();
	static ArrayList<Double[]> fourFoldTestData = new ArrayList<>();
	static ArrayList<Double[]> fullTrainData = new ArrayList<>();
	static ArrayList<Double[]> devData = new ArrayList<>();
	static ArrayList<Double[]> testData = new ArrayList<>();
	static int DIMENSIONALITY = 69; 
	static DecimalFormat df = new DecimalFormat("###.0");
	
	public static void main(String[] args) {
		String fullTrainFilePath = args[1];
		String fullDevFilePath = args[2];
		String fullTestFilePath = args[3];
		
		System.out.println("============================================");
		System.out.println("MAJORITY BASELINE");
		System.out.println("============================================\n");
		
		buildTrainData(fullTrainFilePath);
		buildDevData(fullDevFilePath);
		buildTestData(fullTestFilePath);
		
		majorityBaselineFullDevAndTest();
	}
	
	/**
	 * Returns vector w using the passed learning rate
	 * @param lRate
	 * @return Weights vector w
	 */
	private static void majorityBaselineFullDevAndTest() {
		
		// Train on full
		int countPosLabel = 0;
		int countNegLabel = 0;
		for(Double[] entry : fullTrainData) {
			double y = entry[0];
			if(y == -1) {
				countNegLabel++;
			}
			else {
				countPosLabel++;
			}
		}
		
		int majorityLabel = 0;
		if(countPosLabel > countNegLabel) {
			majorityLabel = 1;
		}
		else {
			majorityLabel = -1;
		}
		
		// Measure Dev accuracy
		int numOfMistakes = 0;
		for(Double[] entry : devData) {
			double expectedY = entry[0];
			
			if(expectedY != (double) majorityLabel) {
				numOfMistakes++;
			}
		}
		
		double devAccuracy = ((devData.size() - (double)numOfMistakes)/devData.size())* 100.0;
					
		System.out.println("Development Set Accuracy: " + df.format(devAccuracy) + " %");
		
		// Measure Dev accuracy
		numOfMistakes = 0;
		for(Double[] entry : testData) {
			double expectedY = entry[0];

			if(expectedY != (double) majorityLabel) {
				numOfMistakes++;
			}
		}

		double testAccuracy = ((testData.size() - (double)numOfMistakes)/testData.size())* 100.0;

		System.out.println("Test Set Accuracy       : " + df.format(testAccuracy) + " %");
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
}

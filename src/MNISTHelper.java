import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;


public class MNISTHelper {

	public static MNISTDigitTrainingRecord[] loadTrainingData(int numbRecords, String pathToImages, String labelFilename) {
		
		MNISTDigitTrainingRecord[] records = new MNISTDigitTrainingRecord[numbRecords];
		
		ArrayList<Integer> labels = readLabels(numbRecords, labelFilename);
		

		String filename = "";
		//Note: filename has (p + 1) in it...00001.tif stored in position 0
		for(int p = 0; p < records.length; p++) {
			if(p + 1 < 10)
				filename = "0000";
			else if(p + 1 < 100)
				filename = "000";
			else if(p + 1< 1000)
				filename = "00";
			else if (p + 1< 10000)
				filename = "0";
			else
				filename = "";
			records[p] = new MNISTDigitTrainingRecord(pathToImages + filename + (p+1) + ".tif", labels.get(p));
			//System.out.println(filename + (p+1) + ".tif goes with label " + labels.get(p) );
		}
		
		return records;
	}

	private static ArrayList<Integer> readLabels(int numbRecords, String labelFilename) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		TextFileReader reader = new TextFileReader(labelFilename);
		String currLine = reader.nextLine();
		while(currLine != null) {
			list.add(Integer.parseInt(currLine));
			currLine = reader.nextLine();
		}
		return list;
	}
	
	public static void saveToFile(MNISTDigitTrainingRecord[] data, String filename) throws FileNotFoundException, IOException {
		
		File file = new File(filename);
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file));
		oos.writeObject(data);
		oos.flush();
		oos.close();
	}
	
	public static MNISTDigitTrainingRecord[] readFromFile(String filename) {
		MNISTDigitTrainingRecord[] data = null;
		
		try {
			File file = new File(filename);
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
			data = (MNISTDigitTrainingRecord[])ois.readObject();
		} 
		catch(Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		return data;
	}

}

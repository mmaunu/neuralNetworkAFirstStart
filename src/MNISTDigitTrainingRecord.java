import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;

import javax.imageio.ImageIO;


public class MNISTDigitTrainingRecord implements TrainingRecord, Serializable {

	private double[] inputs;
	private double[] outputs;
	private int[] imageData;
	
	public MNISTDigitTrainingRecord(String filename, int label) {
		
		loadInputs( filename );
		outputs = new double[10];
		for(int p = 0; p < outputs.length; p++) {
			outputs[p] = 0.0; 
		}
		outputs[label] = 1.0;
	}
	
	private void loadInputs(String filename) {
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(filename));
			
		}
		catch(IOException e) {
			e.printStackTrace();
			System.out.println(filename);
			System.exit(1);
		}
		int[] imageData = new int[784];
		inputs = new double[784];
		img.getRGB(0, 0, 28, 28, imageData, 0, 28);

		
		//imageData seemed to have -1 for white pixels and -MASSIVE for black/gray pixels
		//Renorming to 1.0 for white and 0.0 for black
		for(int p = 0; p < imageData.length; p++) {
			inputs[p] = imageData[p] > -20 ? 1.0 : 0.0;
		}
		
		
		//left here...was part of testing functionality, but not useful for normal operation
//		int[] temp = new int[784];
//		for(int p = 0; p < temp.length; p++) {
//			temp[p] = inputs[p] < 0.5 ? 0 : Integer.MAX_VALUE;
//		}
//		
//		try {
//			img.setRGB(0, 0, 28, 28, temp, 0, 28);
//			ImageIO.write(img, "jpg", new File("out2.jpg"));
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
	}
	@Override
	public double[] getInputs() {
		return inputs;
	}
	

	@Override
	public double[] getOutputs() {
		return outputs;
	}

}

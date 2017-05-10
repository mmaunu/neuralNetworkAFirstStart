import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;

import javax.imageio.ImageIO;
//import com.github.jaiimageio.plugins.tiff.*;



public class NeuralNet {

	public static ActivationFunction actFunk;
	
	public static DecimalFormat decimalForm = new DecimalFormat("#.######");
	
	private double learningRate;
	private int numIterations;
	
	
	private LayerOfNeurons inputLayer, hiddenLayer, outputLayer;
	
	
	/**
	 * Assuming that there is only one hidden layer with numHiddenNeurons in it.
	 * 
	 */
	public NeuralNet(ActivationFunction af,
					double learningRate,
					int numIterations,
					int numInputs,
					int numHiddenNeurons,
					int numOutputs) {
		
		this.actFunk = af;
		this.actFunk.setLearningRate(learningRate);
		
		this.numIterations = numIterations;
		
		inputLayer = new LayerOfNeurons(numInputs, null);
		hiddenLayer = new LayerOfNeurons(numHiddenNeurons, inputLayer);
		
		outputLayer = new LayerOfNeurons(numOutputs, hiddenLayer);
		inputLayer.setNextLayer(hiddenLayer);
		hiddenLayer.setNextLayer(outputLayer);
	}
	
	
	public void forward(double[] startingInputs) {
		inputLayer.setInputLayerValues(startingInputs);
		hiddenLayer.forward();
		outputLayer.forward();
	}
	
	public void train(TrainingRecord[] trainingData) {
		
		/*  "online" mode for now...for each record:
		 *  1) Get inputs, feed to input layer
		 *  2) Do forward prop to get end outputs
		 *  3) Calculate the error term in the output layer (feed desired
		 *  		output values from trainingData to the output layer
		 *  		neurons).
		 *  4) Call backProp method on each layer, from output to input
		 *  		(non-inclusive on input layer since it has no weights)
		 *  5) Call updateWeights method on each layer
		 */
		for(int pos = 0; pos < trainingData.length; pos++) {
			
			//1
			double[] inputs = trainingData[pos].getInputs();
			inputLayer.setInputLayerValues(inputs);
			
			//2
			hiddenLayer.forward();
			outputLayer.forward();
			
//			System.out.println("\n\nNEW RECORD STARTING" + this);
			
			//3
			double[] desiredOutputs = trainingData[pos].getOutputs();
			outputLayer.calcError(desiredOutputs);
			
			//4
			outputLayer.backProp();
			hiddenLayer.backProp();
			
			//5
			hiddenLayer.updateWeights();
			outputLayer.updateWeights();
			
//			System.out.println("\nAfter updating weights " + this);
			
			
			
		}
	}
	
	public String toString() {
		return inputLayer.toString() + "\n\n" + hiddenLayer.toString() + "\n\n" + outputLayer.toString();
	}
	
	public static void main(String[] args) throws IOException {
/*		
		System.out.println("About to load in a bunch of data...could take a while...");
		MNISTDigitTrainingRecord[] records = null;
		String filename = "MNISTData.dat";
		
//		records = MNISTHelper.loadTrainingData(10000);
//		
//		System.out.println("About to save data to a file");
//		
//		MNISTHelper.saveToFile(records, filename);
//		
//		records = null;
//		System.out.println("Now saved...attempting to reload data from the file...");
		
		records = MNISTHelper.readFromFile(filename);
		
		System.out.println("Data read from the file...testing resuming??");
		
		System.out.println("About to create the net itself...I dunno, boss, we could be here a while...");
		NeuralNet nn = new NeuralNet(new SigmoidFunction(), 0.02, 100, 784, 20, 10);
		
		//NOTE TO SELF...DECREASING THE LEARNING RATE SEEMED TO HELP A LOT...TOO MANY HIDDEN NEURONS WAS BAD
		
		for(int times = 0; times < 10; times++) {
			System.out.println("starting round " + times + " of training with current record set");
			nn.train(records);
		}
		
		int successCounter = 0;
		int totalToTest = 200;
		for(int xTest = 0; xTest < totalToTest; xTest++) {
			int posToTest = xTest;
			nn.forward(records[posToTest].getInputs());
			System.out.print("\nOutput for records " + posToTest + " is: " );
			Neuron[] out = nn.outputLayer.getListOfNeurons();
			int posOfMax = 0;
			for(int p = 0; p < out.length; p++)
				if(out[p].getOutput() > out[posOfMax].getOutput())
					posOfMax = p;
			
			System.out.println(posOfMax);
			
			if( records[posToTest].getOutputs()[posOfMax] > .9 )
				successCounter++;
			
			System.out.println("\nOutputs are: ");
			for(Neuron x : out)
				System.out.print(decimalForm.format(x.getOutput()) + ", ");
			System.out.println("\n");
		}
		
		System.out.println("\nThe percent correct for the prediction is " + (double)successCounter/totalToTest);
*/
		
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
///////////  CUSTOM SEPARATOR!!! OMG, DOUBLE SLASHBOW, ALL THE WAY     /////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////		
		
		/* Training worked with logical operators...So now trying it with an extra input that
		 * should get ignored if the thing can really learn...
		*/
/*		
		
		NeuralNet nn = new NeuralNet(new SigmoidFunction(), 0.5, 100, 3, 5, 1);
		 
		TrainingRecord[] records = new TrainingRecord[8];
		int type = LogicalOperatorTrainingRecordWithExtraSillyInput.AND;
		records[0] = new LogicalOperatorTrainingRecordWithExtraSillyInput(type, 0, 0, 0);
		records[1] = new LogicalOperatorTrainingRecordWithExtraSillyInput(type, 0, 1, 0);
		records[2] = new LogicalOperatorTrainingRecordWithExtraSillyInput(type, 1, 0, 0);
		records[3] = new LogicalOperatorTrainingRecordWithExtraSillyInput(type, 1, 1, 0);	
		records[4] = new LogicalOperatorTrainingRecordWithExtraSillyInput(type, 0, 0, 1);
		records[5] = new LogicalOperatorTrainingRecordWithExtraSillyInput(type, 0, 1, 1);
		records[6] = new LogicalOperatorTrainingRecordWithExtraSillyInput(type, 1, 0, 1);
		records[7] = new LogicalOperatorTrainingRecordWithExtraSillyInput(type, 1, 1, 1);	
		
		for(int i = 0; i < 10; i++)
			nn.train(records);
		
		System.out.println(nn);
		System.out.println(nn.outputLayer.getListOfNeurons()[0].getOutput());
		
		double[] input = {0, 1, 0};
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + ", " + input[2] + " is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));
		
		input[0] = 0.0;
		input[1] = 1.0;
		input[2] = 1.0;
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + ", " + input[2] +" is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));
		
		input[0] = 1.0;
		input[1] = 0.0;
		input[2] = 1.0;
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + ", " + input[2] +" is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));
		
		input[0] = 1.0;
		input[1] = 0.0;
		input[2] = 0.0;
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + ", " + input[2] +" is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));
		
		input[0] = 0.0;
		input[1] = 0.0;
		input[2] = 0.0;
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + ", " + input[2] +" is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));

		
		input[0] = 0.0;
		input[1] = 0.0;
		input[2] = 1.0;
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + ", " + input[2] +" is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));
		
		input[0] = 1.0;
		input[1] = 1.0;
		input[2] = 1.0;
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + ", " + input[2] +" is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));
		
		input[0] = 1.0;
		input[1] = 1.0;
		input[2] = 0.0;
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + ", " + input[2] +" is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));
		
*/
		
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
///////////  CUSTOM SEPARATOR!!! OMG, DOUBLE SLASHBOW, ALL THE WAY     /////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
		
		
		
		
		/* Training worked with logical operators...took about 10000 runs of the data array but
		   it definitely worked.
		*/
/*		
		
		NeuralNet nn = new NeuralNet(new SigmoidFunction(), 0.5, 100, 2, 5, 1);
		 
		TrainingRecord[] records = new TrainingRecord[4];
		int type = LogicalOperatorTrainingRecord.NAND;
		records[0] = new LogicalOperatorTrainingRecord(type, 0, 0);
		records[1] = new LogicalOperatorTrainingRecord(type, 1, 0);
		records[2] = new LogicalOperatorTrainingRecord(type, 0, 1);
		records[3] = new LogicalOperatorTrainingRecord(type, 1, 1);	
		
		for(int i = 0; i < 10000; i++)
			nn.train(records);
		
		System.out.println(nn);
		System.out.println(nn.outputLayer.getListOfNeurons()[0].getOutput());
		
		double[] input = {0, 1};
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + " is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));
		
		input[0] = 1.0;
		input[1] = 1.0;
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + " is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));
		
		input[0] = 1.0;
		input[1] = 0.0;
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + " is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));
		
		input[0] = 0.0;
		input[1] = 0.0;
		nn.forward(input);
		System.out.println("output for " + input[0] + ", " + input[1] + " is: " + decimalForm.format(nn.outputLayer.getListOfNeurons()[0].getOutput()));
		

*/

		
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
///////////  CUSTOM SEPARATOR!!! OMG, DOUBLE SLASHBOW, ALL THE WAY     /////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
		
		/* Some testing along the way...not needed, just here for a "record" of testing
		double[] startingInputValues = {2.0, 1.0};
		nn.forward(startingInputValues);
		
		System.out.println(nn);
		
		double[] dOV = {1};
		nn.outputLayer.calcError(dOV);
		nn.outputLayer.backProp();
		nn.hiddenLayer.backProp();
		System.out.println("\n\n");
		System.out.println(nn.hiddenLayer.getListOfNeurons()[0].getBackPropPDErrorTerm());
		*/
		
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
///////////  CUSTOM SEPARATOR!!! OMG, DOUBLE SLASHBOW, ALL THE WAY     /////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////		
		
		/* Some test code for use with the MNIST data set...
		System.out.println("Starting...");
		MNISTDigitTrainingRecord rec = new MNISTDigitTrainingRecord("00001.tif", 7);
		System.out.println(rec.getOutputs());
		System.out.println("after getting outputs, before inputs");
		double[] blah = rec.getInputs();
		System.out.println("now with more input");
		for(int p = 0; p < blah.length; p++)
			System.out.print(blah[p] + ", ");
		System.out.println("\nlength of input array for MNIST image " + blah.length);
		*/
	}
}

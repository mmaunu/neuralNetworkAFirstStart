
public class Neuron {

	/** Used to generate unique IDs for each neuron. */
	private static int totalNumberOfNeurons = 0;
	
	/** Each neuron has a unique ID associated with it. Used in equals method.
	 */
	private int neuronID;
	
	/** Input Neurons from the previous layer */
	private Neuron[] inputs;
	
	/** The layer that this Neuron is in */
	private LayerOfNeurons neuronLayer;
	
	/** These are the weights for each input neuron plus one extra for a bias.
	 *  The bias has output of 1 (which is then modified by the extra weight).
	 *  To be extra clear, weights.length is equal to inputs.length + 1 and the 
	 *  extra space at the end of the weights array is for the bias.
	 */
	private double[] weights;
	
	/** The output for this neuron is either set manually (if it is an input neuron)
	 *  or is calculated from the activation function and the sumOfWeightedInputs.
	 */
	private double output;
	
	/** The weighted sum of the previous layer's inputs.
	 */
	private double sumOfWeightedInputs;
	
	/** This stores the calculated "delta" term that is used in calculating 
	 *  how to modify the weights when we do back propagation. For the output
	 *  layer, this will simply store (the opposite of the error for this 
	 *  output) * ( the derivative of the activation function evaluated at 
	 *  sumOfWeightedInputs ). For non-output neurons, this value will
	 *  be (the sum of all backPropPDErrorTerms in this layer's next layer) *
	 *  (the derivative of the activation function evaluated at this neuron's
	 *  sumOfWeightedInputs).
	 */
	private double backPropPDErrorTerm;
	
	/** This will store the calculated change in weights for each weight 
	 *  associated with this neuron (when doing back prop). These are stored
	 *  in a variable so that we can calculate the changes for the whole 
	 *  network without updating any weights until the end. After all values
	 *  have been calculated, all neurons will update their weights. This 
	 *  prevents the network from changing while we do back prop.
	 */
	private double[] deltaWeights;
	
	/** This value stores desiredOutputValue - actualOutputValue where 
	 * desiredOutputValue is determined by the training data. Only used
	 * in back prop and only on output neurons.
	 */
	private double outputError;
	
	//in debug mode, weights are not random
	private boolean DEBUG = false;
	
	
	/** If inputs is null, then it is assumed that this is an input Neuron (thus has no inputs itself).
	 * In this case, the parameter output will be used as the output from this input and no calculations
	 * are performed; this will always return the value of output in such a case. However, if the array
	 * of inputs is not null, then the value of output is ignored.
	 * @param inputs
	 */
	public Neuron(Neuron[] inputs, LayerOfNeurons layer) {
		this.inputs = inputs;
		this.neuronLayer = layer;
		this.output = 0;
		if(inputs != null) {
			//note that the extra weight is for the bias...which has an input of 1
			weights = new double[inputs.length + 1];
			deltaWeights = new double[inputs.length + 1];
			//start off with random starting rates...
			for(int p = 0; p < weights.length; p++) {
				//giving us a range of [-1, 1] on the weights
				weights[p] = Math.random() * 2 - 1;
				//trying this tactic...force weights to alternate signs...update...didn't work
				//leaving it here to document the "effort"
//				weights[p] = (neuronID%2==0?1:-1)*Math.random();
				if(DEBUG) {
					weights[p] = 0.5;
				}
			}
		}
		sumOfWeightedInputs = 0;
		
		totalNumberOfNeurons++;
		neuronID = totalNumberOfNeurons;
	}
	
	/**This method actually calculates the output value. Don't call this unless you want
	 * to do the math again. If you just want to access the last calculated value, 
	 * use the getOutput method instead.
	 *
	 * @return sum of the weighted inputs, not the output from the network's 
	 * 				activation function.
	 */
	public double calcSumOfWeightedInputs() {
		
		if(inputs == null) {
			return sumOfWeightedInputs;
		}
		
		double result = 0.0;
		
		for(int pos = 0; pos < inputs.length; pos++)
			result += inputs[pos].getOutput() * weights[pos];
	
		//now add in the bias, which has an input of 1 and a weight in the last position
		//of the array...this last position doesn't exist in the input array (note loop 
		//condition above is over inputs.length, so this weight was left out)
		result += 1*weights[weights.length - 1];
		
		sumOfWeightedInputs = result;
		output = NeuralNet.actFunk.evaluate(result);
		
		return sumOfWeightedInputs;
	}
	
	public double getOutput() {
		return output;
	}
	
	/** In an input layer, you might want to be able to manually set the output value. 
	 * 
	 * @param newOutput
	 */
	public void setOutput(double newOutput) {
		output = newOutput;
	}
	
	public double getSumOfWeightedInputs() {
		return sumOfWeightedInputs;
	}
	
	public double[] getWeights() {
		return weights;
	}
	
	public double getBackPropPDErrorTerm() {
		return backPropPDErrorTerm;
	}
	
	/** Given a desired output value, calculate the error term as 
	 * 			desiredOutputValue - output (the actual output).
	 *  This should only be called on output layer neurons...you'll get an 
	 *  error if you try to call it on any other layers.
	 * @param desiredOutputValue
	 */
	public void calcError(double desiredOutputValue) {
		if(neuronLayer.isOutputLayer())
			this.outputError = desiredOutputValue - this.output;
		else
			throw new IllegalStateException("Can't call setError on non-output neurons.");
	}
	
	
	/** 
	 *  This method calculates the deltaWeights values but does not actually change the weights.
	 *  You must call the updateWeights method to do this.
	 */
	public void backProp() {
		int posOfThisNeuronInLayer = getPosOfNeuronInLayer();
		if(neuronLayer.isInputLayer()) {
			throw new IllegalStateException("How did backProp get called on an input layer??");
		}
		else if(neuronLayer.isOutputLayer()) {
			this.backPropPDErrorTerm = outputError * NeuralNet.actFunk.evaluateDeriv(sumOfWeightedInputs);
		}
		else {
			double sum = 0.0;
			//posOfThisNeuronInLayer is used to get the weight between this neuron and
			//the neurons in the next layer (assumption being that the pos in this layer
			//corresponds to the position in the next layer's neuron's weights array)
//			System.out.println("\nPos of this neuron in layer " + posOfThisNeuronInLayer);
			//note that there should be a next layer since we handled the output layer above
			Neuron[] listOfNextLayerNeurons = neuronLayer.getNextLayer().getListOfNeurons();
			for(int p = 0; p < listOfNextLayerNeurons.length; p++) {
				Neuron neuronFromNextLayer = listOfNextLayerNeurons[p];
//				System.out.println("neur from next backPropPD term " + neuronFromNextLayer.getBackPropPDErrorTerm());
//				System.out.println("Weight corr. to this neur and next neur " + neuronFromNextLayer.getWeights()[posOfThisNeuronInLayer]);
				sum +=  neuronFromNextLayer.getBackPropPDErrorTerm() * 
						neuronFromNextLayer.getWeights()[posOfThisNeuronInLayer];
			}
			this.backPropPDErrorTerm = sum * NeuralNet.actFunk.evaluateDeriv(sumOfWeightedInputs);	
		}
		
		//calculate deltaWeights here...deltaWeight is just backPropPDErrorTerm * outputFromPrevNeuron
		for(int p = 0; p < inputs.length; p++) {
			deltaWeights[p] = this.backPropPDErrorTerm * inputs[p].getOutput() * NeuralNet.learningRate;
		}
		//that didn't get the bias...which has an output of 1
		deltaWeights[deltaWeights.length - 1] = this.backPropPDErrorTerm * 1 * NeuralNet.learningRate;
		
//		if(posOfThisNeuronInLayer == 0)
//			System.out.println(deltaWeights[0] + " was deltaWeight and output");
	}

	

	public void updateWeights() {
		for(int p = 0; p < weights.length; p++)
			weights[p] += deltaWeights[p];
	}
	
	public boolean equals(Object obj) {
		if( !(obj instanceof Neuron) )
			throw new IllegalArgumentException("Can only compare Neurons to Neurons. Input was " + obj);
		
		return this.neuronID == ((Neuron)obj).neuronID;
	}
	
	private int getPosOfNeuronInLayer() {
		int posFound = -1;
		Neuron[] listInLayer = neuronLayer.getListOfNeurons(); 
		for(int pos = 0; pos < listInLayer.length; pos++)
			if(listInLayer[pos].neuronID == this.neuronID) {
				posFound = pos;
				break;
			}
		return posFound;
	}
	
	public String toString() {
		if(inputs == null) {
			return "Input Layer Output: " + output + "\t\t";
		}
		else {
			String s = "Weights [";
			for(int pos = 0; pos < inputs.length; pos++)
				s += NeuralNet.decimalForm.format(weights[pos]) + ", ";
			s += NeuralNet.decimalForm.format(weights[weights.length - 1]);
			s += "]";
			
//				s += inputs[pos].getOutput() + " * " + weights[pos] + " + ";
//			s += " a bias of 1 with weight of " + weights[weights.length - 1];
//			s += " sumOfWeightedInputs is " + sumOfWeightedInputs + " and output from act. func. is " + output;
			
			return s;
		}
	}

}

import java.util.ArrayList;


public class LayerOfNeurons {

	private Neuron[] listOfNeurons;
	
	private LayerOfNeurons prevLayer, nextLayer;
	
	/** If prevLayer is null, then it is assumed that this is the input layer. In that case, inputLayerValues
	 * will be used to initialize the outputs for the input layer. If prevLayer is not null, then each Neuron
	 * in this layer will receive inputs from the previous layer.
	 * @param numOfNeurons 		Number of neurons in this layer.
	 * @param prevLayer			Previous layer. For the one hidden layer, this should be the input layer. For the 
	 * 							output layer, this should be the hidden layer.
	 * @param inputLayerValues	Only used if prevLayer is null, sets output values for the input layer.
	 */
	public LayerOfNeurons(int numOfNeurons, LayerOfNeurons prevLayer) {
		listOfNeurons = new Neuron[numOfNeurons];
		
		this.prevLayer = prevLayer;
		
		
		for(int pos = 0; pos < listOfNeurons.length; pos++) {
			if(prevLayer != null)
				listOfNeurons[pos] = new Neuron(prevLayer.getListOfNeurons(), this);
			else
				listOfNeurons[pos] = new Neuron(null, this);
		}
	}
	
	public LayerOfNeurons getPreviousLayer() {
		return prevLayer;
	}
	
	public void setNextLayer(LayerOfNeurons next) {
		nextLayer = next;
	}
	
	public LayerOfNeurons getNextLayer() {
		return nextLayer;
	}

	public Neuron[] getListOfNeurons() {
		return listOfNeurons;
	}
	
	public int getNumberOfNeurons() {
		return listOfNeurons.length;
	}
	
	/** To be used on input layers only...sets the initial input,
	 * which are outputs from the input neurons.
	 * @param outputs
	 */
	public void setInputLayerValues(double[] outputs ) {
		if(prevLayer != null)
			throw new IllegalStateException("You can't set input layer values on a non-input layer.");
		
		if(outputs.length != this.getNumberOfNeurons())
			throw new IllegalArgumentException("Number of input layer values must be equal to the number " +
						"of neurons in the input layer. The param has length " + outputs.length + 
						" and the number of neurons in the input layer is " + getNumberOfNeurons());
		
		for(int pos = 0; pos < outputs.length; pos++)
			listOfNeurons[pos].setOutput(outputs[pos]);
					
	}
	
	public void forward() {
		for(Neuron n : listOfNeurons)
			n.calcSumOfWeightedInputs();
	}
	
	public boolean isInputLayer() {
		return prevLayer == null;
	}
	
	public boolean isOutputLayer() {
		return nextLayer == null;
	}
	
	public void calcError(double[] desiredOutputValues) {
		if( ! isOutputLayer() )
			throw new IllegalStateException("Can't call calcError on " +
						"any layer except the output layer" );
		
		if(desiredOutputValues.length != this.getNumberOfNeurons())
			throw new IllegalArgumentException("Number of desired output values " +
					"must match the number of neurons in the output layer. " +
					"Number of output neurons " + getNumberOfNeurons() + " and " +
					"number of desired output values " + desiredOutputValues.length);
		
		for(int p = 0; p < desiredOutputValues.length; p++) {
			listOfNeurons[p].calcError(desiredOutputValues[p]);
		}
	}
	
	public void backProp() {
		if(isInputLayer())
			throw new IllegalStateException("Can't call backProp on an input layer");
		
		for(Neuron n : listOfNeurons)
			n.backProp();
	}
	
	public void updateWeights() {
		if(isInputLayer())
			throw new IllegalStateException("Can't call updateWeights on an input layer");
		
		for(Neuron n : listOfNeurons)
			n.updateWeights();
	}
	
	public String toString() {
		String s = "";
		for(Neuron n : listOfNeurons)
			s += n + "\t";
		return s;
	}
}

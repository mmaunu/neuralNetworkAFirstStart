
public class LogicalOperatorTrainingRecordWithExtraSillyInput implements TrainingRecord {

	private double[] inputs;
	private double[] outputs;
	
	public static final int OR = 1, 
							AND = 2,
							XOR = 3,
							NAND = 4;
	
	public LogicalOperatorTrainingRecordWithExtraSillyInput(int type, int input1, int input2, int input3) {
		inputs = new double[3];
		inputs[0] = input1;
		inputs[1] = input2;
		inputs[2] = input3;
		
		outputs = new double[1];
		
		if(type == OR)
			outputs[0] = input1 == 1 || input2 == 1 ? 1.0 : 0.0;
		else if(type == AND)
			outputs[0] = input1 == 1 && input2 == 1 ? 1.0 : 0.0;
		else if(type == XOR)
			outputs[0] = input1 != input2 ? 1.0 : 0.0;
		else if(type == NAND)
			outputs[0] = input1 == 0 || input2 == 0 ? 1.0 : 0.0;
		else
			throw new IllegalArgumentException("The type parameter to this constructor must be one " +
						"of the class constants OR, AND, XOR, or NAND.");
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

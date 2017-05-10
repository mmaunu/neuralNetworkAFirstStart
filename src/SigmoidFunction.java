
public class SigmoidFunction implements ActivationFunction{

	private double learningRate = 1.0;
	
	public void setLearningRate(double rate) {
		learningRate = rate;
	}
	
	public double evaluate(double input) {
		return 1.0/(1.0 + Math.exp(-input*learningRate));
	}
	
	public double evaluateDeriv(double input) {
		return evaluate(input) * (1 - evaluate(input));
	}
}

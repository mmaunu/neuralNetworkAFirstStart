
public interface ActivationFunction {

	public double evaluate(double input);
	public double evaluateDeriv(double input);
	public void setLearningRate(double learningRate);
	
}

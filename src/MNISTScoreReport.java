
public class MNISTScoreReport implements Comparable {
	private double learningRate;
	private int numbHiddenNeurons;
	private double percentCorrect;
	
	public MNISTScoreReport(double lr, int numbHidden, double percent) {
		learningRate = lr;
		numbHiddenNeurons = numbHidden;
		percentCorrect = percent;
	}

	@Override
	public int compareTo(Object obj) {
		
		if(this.percentCorrect > ((MNISTScoreReport)obj).percentCorrect)
			return 1;
		else if (this.percentCorrect < ((MNISTScoreReport)obj).percentCorrect)
			return -1;
		else
			return 0;
	}
	
	public String toString() {
		return  "Learning Rate: " + learningRate + "\n" +
				"Number of Hidden Neurons: " + numbHiddenNeurons + "\n" +
				"Percent Correct: " + percentCorrect + "\n";
	}
	
}

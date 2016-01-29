import java.io.*;
import java.util.Random;

/**
 * This is the class for an extremely simple learning algorithm that
 * finds the most frequent class in the training data, and then
 * predicts that each new test example belongs to this class.
 */
public class NeuralNetClassifier implements Classifier {

    private double rate;
    private double epochs;
    
	private int N;
	private int numAttrs;
	private NumericDataSet d;
	private double[] w;
	private String author = "Advait Chauhan";
	private String description = "Single Layer Neural Network (from RN 18.7) with learning rate " + rate + " and " + epochs + " epochs.";

	/**
	 * This constructor takes as input a dataset and computes and
	 * stores the most frequent class
	 */
	public NeuralNetClassifier(NumericDataSet d, double frac_test, int epochs, double rate) {
		Random rand = new Random();
		this.numAttrs = d.numAttrs;
		this.d = d;
		this.epochs = epochs;
		this.rate = rate;
		
		double temp = d.numTrainExs * frac_test;
		this.N = (int) temp;
		
		w = new double[d.numAttrs];
		double y = 0;
		
		// initialize edge weights
		for (int i = 0; i < w.length; i++) {
			w[i] = rand.nextDouble()/200.0;
		}
		
		// run epochs
		for (int iter = 0; iter < epochs; iter++)
        {
			// go through each training example
			for (int i = 0; i < N; i++)
            {
				// propogate network forward to calculate hypothesized label
				double inJ = 0;
				for (int k = 0; k < w.length; k++)
                {
					inJ += w[k]*((double) d.trainEx[i][k]);
				}
				y = threshold(inJ);
                
                //propogate the error backwards
				//double delta = (d.trainLabel[i]-y)*derivThreshold(inJ);
				
				double delta = d.trainLabel[i]-y;
				for (int j = 0; j < w.length; j++)
                {
					w[j] += rate*delta*d.trainEx[i][j];
				}
			}
		}
		
		double acc = predictTraining();
		double acc2 = predictTesting();
		System.out.println("Training accuracy: " + acc);
		System.out.println("Testing accuracy: " + acc2);
	}
	
	/*sigmoid function*/
	private double threshold(double x)
    {
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	/*derivative of sigmoid function*/
	private double derivThreshold(double x)
    {
		return threshold(x)*(1-threshold(x));
	}

	/** The prediction method ignores the given example and predicts
	 * with the most frequent class seen during training.
	 */
	public int predict(int[] ex)
    {
		double x = 0;
		for (int i = 0; i < numAttrs; i++)
        {
			x += w[i]*((double)ex[i]);
		}
		return (int) Math.round(threshold(x));
	}
	
	public double predictTraining()
    {
		int numCorrect = 0;
		for (int i = 0; i < N; i++)
        {
			if (d.trainLabel[i] == predict(d.trainEx[i])) numCorrect++;
		}
		return numCorrect/(double)N;
	}
	
	public double predictTesting()
	{
		int numCorrect = 0;
		for (int i = N; i < d.numTrainExs; i++)
		{
			if (d.trainLabel[i] == predict(d.trainEx[i]))
				numCorrect++;
		}
		return numCorrect/(double)(d.numTrainExs-N);
	}
	

	/** This method returns a description of the learning algorithm. */
	public String algorithmDescription()
    {
		return description;
	}

	/** This method returns the author of this program. */
	public String author()
    {
		return author;
	}

	/** A simple main for testing this algorithm.  This main reads a
	 * filestem from the command line, runs the learning algorithm on
	 * this dataset, and prints the test predictions to filestem.testout.
	 */
	public static void main(String argv[]) throws FileNotFoundException, IOException {

		// if (argv.length < 1) {
		// 	System.err.println("argument: filestem");
		// 	return;
		// }

		// String filestem = argv[0];
		// NumericDataSet d = new NumericDataSet(filestem);
		// Classifier c = new NeuralNetClassifier(d, .625);
		// d.printTestPredictions(c, filestem);

		/*INPUTS TO MODIFY*/
        double p = 5.0/8;	//fraction of data to train with
        int k = 10000;		//number of epochs
        double l = .01;		//learning rate

        System.out.println("----Test with " + k + " epochs and learning rate " + l + "-----");
        System.out.println("Census");
    	NumericDataSet census = new NumericDataSet("data/census");
    	Classifier censusPredict = new NeuralNetClassifier(census, p, k, l);
    	census.printTestPredictions(censusPredict, "data/census");

        System.out.println("OCR 49");
        NumericDataSet ocr49 = new NumericDataSet("data/ocr49");
        NeuralNetClassifier ocr49Predict = new NeuralNetClassifier(ocr49, p, k, l);
        ocr49.printTestPredictions(ocr49Predict, "data/ocr49");

        System.out.println("OCR 17");
        NumericDataSet ocr17 = new NumericDataSet("data/ocr17"); 
        NeuralNetClassifier ocr17Predict = new NeuralNetClassifier(ocr17, p, k, l);   
        ocr17.printTestPredictions(ocr17Predict, "data/ocr17");


        System.out.println("DNA");
        NumericDataSet dna = new NumericDataSet("data/dna"); 
        NeuralNetClassifier dnaPredict = new NeuralNetClassifier(dna, p, k, l);   
        dna.printTestPredictions(dnaPredict, "data/dna");

	}
}

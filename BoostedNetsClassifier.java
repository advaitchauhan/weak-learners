import java.util.*;
import java.io.*;

public class BoostedNetsClassifier implements Classifier
{
    private String author = "Advait Chauhan";
    private String description = "Neural Nets with Adaboost";
    
    private double rate = .1;
    private double epochs = 1;

    private NumericDataSet ds;


    private int N; //number of training examples
    private int A; //number of attributes
    private int T; //rounds of boosting
    private double [][] hyps; //boosting hypotheses
    private double [] hwts; //boosting hypotheses weights


    /* Builds a single layer neural net hypotheses where examples are weighted by the adaboost example weights, returns
       the array of neural net weights that define the net*/
    private double [] NetClassifier(double [] weights)
    {	
    	
    	/* SAMPLE EXAMPLES BY WEIGHT: choose N of the training examples on each round of 
        boosting by sampling with replacement according to the weight distribution.*/
    	int [] [] selectExamples = new int [N][A];
    	int [] selectLabel = new int [N];

    	for (int n = 0; n < N; n++)
    	{
    		int sampledIndex = BoostedNetsClassifier.sampleFromDistribution(weights);
    		selectLabel[n] = ds.trainLabel[sampledIndex];
    		for (int i = 0; i < A; i++)
    		{
    			selectExamples[n][i] = ds.trainEx[sampledIndex][i];
    		}
    	}

        double[]  w = new double[A];
        double y = 0;

        // initialize edge weights
        Random rand = new Random();
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
                    inJ += w[k]*((double) selectExamples[i][k]);
                }
                y = threshold(inJ);
                
                //propogate the error backwards
                //double delta = (d.trainLabel[i]-y)*derivThreshold(inJ);
                double delta = selectLabel[i]-y;
                for (int j = 0; j < w.length; j++)
                {
                    w[j] += rate*delta*selectExamples[i][j];
                }
            }
        }

        return w;

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

    /*returns normalized version of double array*/
    private static double[] normalize(double [] input)
    {
        double [] ret = new double[input.length];
        double sum = 0;
        for (double d: input)
            sum += d;
        for (int i = 0; i < ret.length; i++)
        {
            ret[i] = input[i] / sum;
        }
        return ret;
    }

    /*intervalizes a normalized probability*/
    private static double [] discretize(double [] input)
    {
        double [] ret = new double[input.length];
        ret[0] = input[0];
        for (int i = 1; i < ret.length; i++)
        {
            ret[i] = ret[i-1] + input[i];
        }
        return ret;
    }

    /*given a normalized probability distribution, randomly samples
      and returns one of the values*/
    private static int sampleFromDistribution(double [] dist)
    {
    	Random random = new Random();
		double [] intervals = BoostedNetsClassifier.discretize(dist); 
		double rand = random.nextDouble();
		int sampled_value = -1;

        for (int i = 0; i < intervals.length; i++)
        {
            if (rand < intervals[i])
            {
                sampled_value = i;
                break;
            }
        }
        return sampled_value; 	
    }

    /*Given the weights for each attribute in a single layer
     neural net, return the computed prediction*/
    private int predictNet(int [] ex, double [] w)
    {
        double x = 0;
        for (int i = 0; i < A; i++)
        {
            x += w[i]*((double)ex[i]);
        }
        return (int) Math.round(threshold(x));
    }

/******************************************
/ PUBLIC METHODS
/******************************************/
    public BoostedNetsClassifier(NumericDataSet d, int numBoosts, double train_frac)
    {
        ds = d;
        double temp = ds.numTrainExs * train_frac;
        N = (int) temp;
        A = ds.numAttrs;
        T = numBoosts;

    	double [] ewts = new double[N]; //example weights

        hwts = new double [T]; //hypotheses weights
    	hyps = new double [T][]; //storage of T neural net hypotheses

        /*initializing weights too all be equal*/
        for (int i = 0; i < ewts.length; i++)
    	{
    		ewts[i] = (double) 1 / N;
    	}

        /*generation of hypotheses by the Adaboost algorithm in Schapire 1.1*/
        for (int t = 0; t < T; t++)
        {
            // System.out.println("Boosting Round: " + t);
            /*generate neural net hypothesis based on weights*/
            hyps[t] = NetClassifier(ewts);
            
            /*determine error of hypothesis on training data*/
            boolean [] correct = new boolean[N];
            int numRight = 0;
            double err = 0;

            for (int e = 0; e < N; e++)
            {
                if (predictNet(ds.trainEx[e], hyps[t]) == ds.trainLabel[e])
                {
                    correct[e] = true;
                    numRight++;
                }
                else
                {
                    correct[e] = false;
                    err += ewts[e];
                }
            }

            /*compute hypothesis weight*/
            if (err == 0.0)
                hwts[t] = 0;
            else
                hwts[t] = .5 * Math.log((1-err)/err);

            /*compute example weights for next round of boosting*/
            for (int e = 0; e < N; e++)
            {
                if (correct[e])
                    ewts[e] *= Math.pow(Math.E, -1*hwts[t]);
                else
                    ewts[e] *= Math.pow(Math.E, hwts[t]);
            }

            ewts = BoostedNetsClassifier.normalize(ewts);

            err = 0;
            for (int e = 0; e < N; e++)
            {
                if (predictNet(ds.trainEx[e], hyps[t]) == ds.trainLabel[e])
                {
                    correct[e] = true;
                    numRight++;
                }
                else
                {
                    correct[e] = false;
                    err += ewts[e];
                }
            }
        }
    }


    /** A method for predicting the label of a given example <tt>ex</tt>
     * represented, as in the rest of the code, as an array of values
     * for each of the attributes.  The method should return a
     * prediction, i.e., 0 or 1.
     */
    public int predict(int[] ex)
    {
        double sum = 0;
        for (int t = 0; t < T; t++)
        {
            int pred = predictNet(ex, hyps[t]);
            if (pred == 0) pred = -1; //we temporarily assign a value of -1 to the 0 prediction
            sum += pred*hwts[t];
        }

        if (sum < 0)
            return 0; //reassign value of 0 to negative prediction
        else
            return 1;
    }

    public double predictTraining()
    {
        int numCorrect = 0;
        for (int e = 0; e < N; e++)
        {
            int pred = predict(ds.trainEx[e]);
            if (pred == ds.trainLabel[e]) numCorrect++;
        }
        return (double)numCorrect / N;
    }

    public double predictTesting()
    {
        int numCorrect = 0;
        int numTests = ds.numTrainExs - N;
        for (int e = N; e < ds.numTrainExs; e++)
        {
            int pred = predict(ds.trainEx[e]);
            if (pred == ds.trainLabel[e]) numCorrect++;
        }
        return (double)numCorrect / numTests;
    }

    public String algorithmDescription()
    {
    	return description;
    }

    public String author()
    {
    	return author;
    }

    public static void main (String [] args) throws FileNotFoundException, IOException

    {
        double p = 8.0/10;
        int k = 1000; //number of rounds of boosting
        System.out.println("----Adaboosted Neural Nets with " + k + " rounds of boosting and " + p + " of data trained on");
        System.out.println("Census");
    	NumericDataSet census = new NumericDataSet("data/census");
    	BoostedNetsClassifier censusPredict = new BoostedNetsClassifier(census, k, p);
    	census.printTestPredictions(censusPredict, "data/census");

        System.out.println("OCR 49");
        NumericDataSet ocr49 = new NumericDataSet("data/ocr49");
        BoostedNetsClassifier ocr49Predict = new BoostedNetsClassifier(ocr49, k, p);
        ocr49.printTestPredictions(ocr49Predict, "data/ocr49");

        System.out.println("OCR 17");
        NumericDataSet ocr17 = new NumericDataSet("data/ocr17"); 
        BoostedNetsClassifier ocr17Predict = new BoostedNetsClassifier(ocr17, k, p);   
        ocr17.printTestPredictions(ocr17Predict, "data/ocr17");


        System.out.println("DNA");
        NumericDataSet dna = new NumericDataSet("data/dna"); 
        BoostedNetsClassifier dnaPredict = new BoostedNetsClassifier(dna, k, p);           dna.printTestPredictions(dnaPredict, "data/dna");

    }

}
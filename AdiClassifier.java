import java.util.*;
import java.io.*;

public class AdiClassifier implements Classifier
{
    private String author = "Advait Chauhan";
    private String description = "Adaboost on Decision Stumps, in which to generate decision stumps example data is randomly sampled by Adaboost weights and then the entropy minimizing stump is selected";
    private DiscreteDataSet ds;
    private int [] stump; 
    private int N; //number of training examples
    private int A; //number of attributes
    private int T; //rounds of boosting
    private int [][] hyps; //boosting hypotheses
    private double [] hwts; //boosting hypotheses weights


    /* Builds a decision stump based on the training data and provided weights. Outputs stump 
    as an array where the 1st element tells us the ATTRIBUTE NUMBER that the stump tests, and then 
    the remaining values map the kth attribute value to it's prediction*/
    private int [] stumpClassifier(double [] weights)
    {	
    	
    	/* SAMPLE EXAMPLES BY WEIGHT: choose N of the training examples on each round of 
        boosting by sampling with replacement according to the weight distribution.*/
    	int [] [] selectExamples = new int [N][A];
    	int [] selectLabel = new int [N];

    	for (int n = 0; n < N; n++)
    	{
    		int sampledIndex = AdiClassifier.sampleFromDistribution(weights);
    		selectLabel[n] = ds.trainLabel[sampledIndex];
    		for (int i = 0; i < A; i++)
    		{
    			selectExamples[n][i] = ds.trainEx[sampledIndex][i];
    		}
    	}

    	/*CHOOSE THE ENTROPY MINIMIZING ATTRIBUTE*/
    	int bestAttrib = -1;
    	double minEntropy = Double.POSITIVE_INFINITY;
    	for (int a = 0; a < A; a++)
    	{
    		double curEntropy = attributeEntropy(a, selectExamples, selectLabel);
    		if (curEntropy < minEntropy)
    		{
    			minEntropy = curEntropy;
    			bestAttrib = a;
    		}
    	}
 
    	/*for each of bestatrib's values, determine the majority classification*/
        int numAttribVals = ds.attrVals[bestAttrib].length;
        int [] ret = new int[1 + numAttribVals];
        ret[0] = bestAttrib;

    	for (int k = 0; k < ds.attrVals[bestAttrib].length; k++) 
    	{
    		int numZeros = 0;
    		int numOnes = 0;
    		for (int e = 0; e < N; e++)
    		{
    			if (selectExamples[e][bestAttrib] == k)
    			{
    				if (selectLabel[e] == 0)
    					numZeros++;
    				else
    					numOnes++;
    			}
    		}

    		ret[k+1] = (numZeros > numOnes ? 0 : 1); //we arbitrarily give ties to 1
    	}

    	return ret;
    }

    /*returns the expected entropy if attribute a is tested, given the inputted training data*/
    private double attributeEntropy(int a, int [][] selectExamples, int [] selectLabel)
    {
        double curEntropy = 0;
        for (int k = 0; k < ds.attrVals[a].length; k++)
        {
            /*compute likelihood and remaining entropy of value k for attribute a*/
            int numK = 0;
            int numK_positive = 0;
            for (int e = 0; e < N; e++)
            {
                if (selectExamples[e][a] == k) 
                {
                    numK++;
                    if (selectLabel[e] == 0) numK_positive++;
                }
            }

            double probK = (double)numK / N; //likelihood of value k for attribute a
            
            if (probK == 0) //if likelihood is 0, there is no addition to entropy
                continue;

            double entK = AdiClassifier.entropy((double)numK_positive / numK); //entropy of examples with value k for attribute a
            curEntropy += probK*entK;
        }
        return curEntropy;
    }

    /*entropy for a boolean random variable with probability parameter p*/
    private static double entropy(double p)
    {
    	if (p == 0) return 0;
    	else if (p == 1) return 0;
    	else return -1*(p*(Math.log(p)/Math.log(2)) + (1-p)*(Math.log(1-p)/Math.log(2)));
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
		double [] intervals = AdiClassifier.discretize(dist); 
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

    private int predictStump(int [] ex, int [] stump)
    {
        int bestAttribVal = ex[stump[0]];
        return stump[bestAttribVal+1];
    }

    public double predictTraining()
    {
        int numCorrect = 0;
        for (int e = 0; e < N; e++)
        {
            // System.out.println("Example#: " + e);
            int pred = predict(ds.trainEx[e]);

            // System.out.println(ds.trainLabel[e] + " vs. " + pred);
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

/******************************************
/ PUBLIC METHODS
/******************************************/
    public AdiClassifier(DiscreteDataSet d, int numBoosts, double train_frac)
    {
        ds = d;
        double temp = ds.numTrainExs * train_frac;
        N = (int) temp;
        A = ds.numAttrs;
        T = numBoosts;

    	double [] ewts = new double[N]; //example weights

        hwts = new double [T]; //hypotheses weights
    	hyps = new int [T][]; //storage of T stump hypotheses

        /*initializing weights too all be equal*/
        for (int i = 0; i < ewts.length; i++)
    	{
    		ewts[i] = (double) 1 / N;
    	}
        // System.out.println();

        /*generation of hypotheses by the Adaboost algorithm in Schapire 1.1*/
        for (int t = 0; t < T; t++)
        {
            //System.out.println("Boosting Round: " + t);
            /*generate decision stump hypothesis based on weights*/
            hyps[t] = stumpClassifier(ewts);
            // System.out.println("Attribute tested: " + hyps[t][0]);
            
            /*determine error of hypothesis on training data*/
            boolean [] correct = new boolean[N];
            int numRight = 0;
            double err = 0;

            for (int e = 0; e < N; e++)
            {
                if (predictStump(ds.trainEx[e], hyps[t]) == ds.trainLabel[e])
                {
                    correct[e] = true;
                    numRight++;
                }
                else
                {
                    correct[e] = false;
                    err += ewts[e];
                }

                // System.out.print(correct[e] + " ");
            }

            // System.out.println();

            // System.out.println("Accuracy: " + numRight/(double)N);
            // System.out.printf("Weighted Error: %.3f", err);
            // System.out.println();

            /*compute hypothesis weight*/
            hwts[t] = .5 * Math.log((1-err)/err);
            // System.out.println("Alpha: " + hwts[t]);


            /*compute example weights for next round of boosting*/
            for (int e = 0; e < N; e++)
            {
                if (correct[e])
                    ewts[e] *= Math.pow(Math.E, -1*hwts[t]);
                else
                    ewts[e] *= Math.pow(Math.E, hwts[t]);
            }

            ewts = AdiClassifier.normalize(ewts);

            err = 0;
            for (int e = 0; e < N; e++)
            {
                if (predictStump(ds.trainEx[e], hyps[t]) == ds.trainLabel[e])
                {
                    correct[e] = true;
                    numRight++;
                }
                else
                {
                    correct[e] = false;
                    err += ewts[e];
                }

                // System.out.print(correct[e] + " ");
            }
            // System.out.println("Error with respect to new weights is: " + err);

            // for (int e = 0; e < N; e++)
            // {
            //     System.out.printf("%.3f  ", ewts[e]);
            // }
            // System.out.println();
            // System.out.println();

        }


        // double acc1 = predictTraining();
        // System.out.println("Accuracy on Training Data: " + acc1);
        // double acc2 = predictTesting();
        // System.out.println("Accuracy on Test Data: " + acc2);
        // System.out.println();

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
            int pred = predictStump(ex, hyps[t]);
            if (pred == 0) pred = -1; //we temporarily assign a value of -1 to the 0 prediction
            sum += pred*hwts[t];
        }

        if (sum < 0)
            return 0; //reassign value of 0 to negative prediction
        else
            return 1;
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
        double p = 5.0/8;
        int k = 100; //number of rounds of boosting
        System.out.println("----Test with " + k + " rounds of boosting and " + p + " of data trained on");
        System.out.println("Census");
    	DiscreteDataSet census = new DiscreteDataSet("data/census");
    	AdiClassifier censusPredict = new AdiClassifier(census, k, p);
    	census.printTestPredictions(censusPredict, "data/census");

        System.out.println("OCR 49");
        DiscreteDataSet ocr49 = new DiscreteDataSet("data/ocr49");
        AdiClassifier ocr49Predict = new AdiClassifier(ocr49, k, p);
        ocr49.printTestPredictions(ocr49Predict, "data/ocr49");

        System.out.println("OCR 17");
        DiscreteDataSet ocr17 = new DiscreteDataSet("data/ocr17"); 
        AdiClassifier ocr17Predict = new AdiClassifier(ocr17, k, p);   
        ocr17.printTestPredictions(ocr17Predict, "data/ocr17");


        System.out.println("DNA");
        DiscreteDataSet dna = new DiscreteDataSet("data/dna"); 
        AdiClassifier dnaPredict = new AdiClassifier(dna, k, p);   
        dna.printTestPredictions(dnaPredict, "data/dna");

    }

}
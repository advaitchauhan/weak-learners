import java.util.*;
import java.io.*;

public class AdiClassifier2 implements Classifier
{
    private String author = "Advait Chauhan";
    private String description = "Adaboost on Decision Stumps where the stumps directly minimize weighted training error";
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

    	/*determine predictions for each attribute*/
        int [][] ret = new int[A][];
        for (int a = 0; a < A; a++)
        {
            /*for each of attribute a's values, determine the majority classification*/
            int numAttribVals = ds.attrVals[a].length;
            int [] retA = new int [1 + numAttribVals];
            retA[0] = a;
        	for (int k = 0; k < ds.attrVals[a].length; k++) 
        	{
        		int numZeros = 0;
        		int numOnes = 0;
        		for (int e = 0; e < N; e++)
        		{
        			if (ds.trainEx[e][a] == k)
        			{
        				if (ds.trainLabel[e] == 0)
        					numZeros++;
        				else
        					numOnes++;
        			}
        		}

        		retA[k+1] = (numZeros > numOnes ? 0 : 1); //we arbitrarily give ties to 1
        	}
            ret[a] = retA;
        }

        /*determine the attribute whose stump prediction on training data gives us minimum weighted error*/
        int bestAttrib = -1;
        double minWeightedError = Double.POSITIVE_INFINITY;
        for (int a = 0; a < A; a++)
        {
            double err = 0;
            for (int e = 0; e < N; e++)
            {
                if (predictStump(ds.trainEx[e], ret[a]) != ds.trainLabel[e])
                    err += weights[e];
            }

            if (err < minWeightedError)
            {
                bestAttrib = a;
                minWeightedError = err;
            }
        }  

    	return ret[bestAttrib];
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
            int pred = predict(ds.trainEx[e]);
            if (pred == ds.trainLabel[e]) numCorrect++;
        }
        return (double)numCorrect / N;
    }

    public double predictTesting()
    {
        int numCorrect = 0;
        for (int e = N; e < ds.numTrainExs; e++)
        {
            int pred = predict(ds.trainEx[e]);
            if (pred == ds.trainLabel[e]) numCorrect++;
        }
        return (double) numCorrect / (ds.numTrainExs - N);
    }

/******************************************
/ PUBLIC METHODS
/******************************************/
    public AdiClassifier2(DiscreteDataSet d, int numBoosts, double train_frac)
    {
        ds = d;
        double temp = train_frac * ds.numTrainExs;
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

        /*generation of hypotheses by the Adaboost algorithm in Schapire 1.1*/
        for (int t = 0; t < T; t++)
        {
            // System.out.println("Boosting Round: " + t);
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
            }

            // System.out.println("Accuracy: " + numRight/(double)N);
            // System.out.printf("Weighted Error: %.3f", err);
            // System.out.println();

            /*compute hypothesis weight*/
            hwts[t] = .5 * Math.log((1-err)/err);
            // System.out.println("Alpha: " + hwts[t]);
            // System.out.println();


            /*compute example weights for next round of boosting*/
            for (int e = 0; e < N; e++)
            {
                if (correct[e])
                    ewts[e] *= Math.pow(Math.E, -1*hwts[t]);
                else
                    ewts[e] *= Math.pow(Math.E, hwts[t]);
            }

            ewts = AdiClassifier2.normalize(ewts);
        }

        // double acc = predictFullTraining();
        // System.out.println("Accuracy on Training Data: " + acc);
        // double acc2 = predictFullTesting();
        // System.out.println("Accuracy on Testing Data: " + acc2);
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
        int k = 1000; //number of rounds of boosting
        System.out.println("----Test with " + k + " rounds of boosting and " + p + " of data trained on");
        System.out.println("Census");
        DiscreteDataSet census = new DiscreteDataSet("/Users/advaitchauhan/Documents/Fall2015/COS402/ML/data/census");
        AdiClassifier2 censusPredict = new AdiClassifier2(census, k, p);
        census.printTestPredictions(censusPredict, "/Users/advaitchauhan/Documents/Fall2015/COS402/ML/data/census");

        System.out.println("OCR 49");
        DiscreteDataSet ocr49 = new DiscreteDataSet("/Users/advaitchauhan/Documents/Fall2015/COS402/ML/data/ocr49");
        AdiClassifier2 ocr49Predict = new AdiClassifier2(ocr49, k, p);
        ocr49.printTestPredictions(ocr49Predict, "/Users/advaitchauhan/Documents/Fall2015/COS402/ML/data/ocr49");

        System.out.println("OCR 17");
        DiscreteDataSet ocr17 = new DiscreteDataSet("/Users/advaitchauhan/Documents/Fall2015/COS402/ML/data/ocr17"); 
        AdiClassifier2 ocr17Predict = new AdiClassifier2(ocr17, k, p);   
        ocr17.printTestPredictions(ocr17Predict, "/Users/advaitchauhan/Documents/Fall2015/COS402/ML/data/ocr17");


        System.out.println("DNA");
        DiscreteDataSet dna = new DiscreteDataSet("/Users/advaitchauhan/Documents/Fall2015/COS402/ML/data/dna"); 
        AdiClassifier2 dnaPredict = new AdiClassifier2(dna, k, p);   
        dna.printTestPredictions(dnaPredict, "/Users/advaitchauhan/Documents/Fall2015/COS402/ML/data/dna");
    }
}
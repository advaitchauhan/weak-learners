For an assignment in my artificial intelligence class, we had to implement a classifier. I ended up building several because I wanted to analyze how different weak learners would perform with Adaboost. 

I first implemented Adaboost with decision stumps by following the Adaboost algorithm from Schapire, and choosing the decision stump for each round of boosting that would minimize the weighted training error. The implementation is in AdiClassifier2.java.

I was then curious if using Adaboost, but selecting the stump hypothesis by finding the entropy minimizing attribute test (entropy defined in R/N 18.3.4), rather than directly minimizing weighted training error, would yield different/better predictions. I implemented the attribute test by finding the attribute that maximized the expected reduction in entropy when tested (exactly following R/N 18.3.4). To incorporate the weights on the training examples, I trained on a random subset of the training examples on each round of boosting by resampling according to the example weight distribution. The implementation is in AdiClassifier.java.

Next, I implemented neural nets as per the backwards-propagation algorithm in R&N, with a couple of adjustments, and only creating a single layer. Through systematic testing for efficacy, I found that removing the derivative of the sigmoid function from the delta calculation vastly improved the percentage of correct guesses. I wasn't sure about why this was, but found that single layer neural net algorithms online (even those using sigmoid functions) had sometimes omitted the derivative from the calculation too. Lastly, I set set the initial edge weights to random values between 0 and 0.005. After some testing testing, I found this to be better than initializing them to 0. The implementation is contained in NeuralNetsClassifier.java.

Finally, having implemented the neural net, I decided to try using it as a weak learning hypothesis and  feeding it into Adaboost. To make the neural net learner weak, I made it run for only 1 epoch and used a heavy learning rate of .1 here (also note that the network being single layer also helps make it weak). I incorporated the example data weights into each generated neural net hypothesis by sampling the example data to feed into neural net according to the given data weight distribution. The implementation is contained in BoostedNeuralNets.java.

References:

*Stuart J. Russell and Peter Norvig(R/N). Artificial Intelligence: A Modern Approach
*Robert E. Schapire and Yoav Freund.Boosting: Foundations and Algorithms.
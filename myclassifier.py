import numpy as np

class SpamClassifier:
        
    def train(self):
        # load training data into array
        training_data = np.loadtxt(open("training_spam.csv"), delimiter=",").astype(np.int)
        
        self.log_class_priors = self.estimate_log_class_priors(training_data)
        self.log_class_conditional_likelihoods = self.estimate_log_class_conditional_likelihoods(training_data)
    
    def estimate_log_class_priors(self, data):
        """
        Given a data set with binary response variable (0s and 1s) in the
        left-most column, calculate the logarithm of the empirical class priors,
        that is, the logarithm of the proportions of 0s and 1s:
            log(p(C=0)) and log(p(C=1))

        :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                     the first column contains the binary response (coded as 0s and 1s).

        :return log_class_priors: a numpy array of length two
        """
        # sum leftmost column and divide by length of column to get probability of spam
        prob_spam = data[:,0].sum() / len(data)
        # P(spam) + P(ham) = 1 so
        prob_ham = 1-prob_spam
        # take logs to reduce risk of precision loss
        log_class_priors = np.array([np.log(prob_spam), np.log(prob_ham)])

        return log_class_priors

    def estimate_log_class_conditional_likelihoods(self, data, alpha=1):
        """
        Given a data set with binary response variable (0s and 1s) in the
        left-most column and binary features (words), calculate the empirical
        class-conditional likelihoods, that is,
        log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

        Assume a multinomial feature distribution and use Laplace smoothing
        if alpha > 0.

        :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]

        :return theta:
            a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
            logarithm of the probability of feature i appearing in a sample belonging 
            to class j.
        """

        # seperate class
        spam = data[data[:, 0] == 1]
        ham = data[data[:, 0] == 0]

        # get number of each in class
        n_spam = len(spam)
        n_ham = len(ham)

        # get number of features
        num_features = len(spam[0])
        # for clarity comparing to pseudocode
        k = num_features

        # get number of spam with each feature, same for ham
        n_spam_w = spam.sum(axis=0)[1:]
        n_ham_w = ham.sum(axis=0)[1:]

        # get total number of occurences of spam and ham words
        n_spam = sum(n_spam_w)
        n_ham = sum(n_ham_w)

        # calc theta
        theta_spam = []
        for n_c_w in n_spam_w:
            # Laplace smoothing
            theta_spam.append(np.log((n_c_w+alpha) / (n_spam + k*alpha)))
        theta_ham = []
        for n_c_w in n_ham_w:
            # Laplace smoothing
            theta_ham.append(np.log((n_c_w+alpha) / (n_ham + k*alpha)))

        return np.array([theta_spam, theta_ham])

    
    def predict(self, new_data):
        """
        Given a new data set with binary features, predict the corresponding
        response for each instance (row) of the new_data set.

        :param new_data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
        :param log_class_priors: a numpy array of length 2.
        :param log_class_conditional_likelihoods: a numpy array of shape = [2, n_features].
            theta[j, i] corresponds to the logarithm of the probability of feature i appearing
            in a sample belonging to class j.
        :return class_predictions: a numpy array containing the class predictions for each row
            of new_data.
        """

        # calc probability times value for each w_i - operand of summation in formula
        # i.e. w_i * theta_c,w_i
        w_theta_spam = new_data * self.log_class_conditional_likelihoods[0]
        w_theta_ham = new_data * self.log_class_conditional_likelihoods[1]

        # to store result
        result = np.zeros(len(new_data))

        # iterate over all spam and ham so can compare which gives a larger value
        for c in range(0, len(w_theta_spam)):
            # calc the two arg values by adding priors to each summation
            args_spam = self.log_class_priors[0] + sum(w_theta_spam[c])
            args_ham = self.log_class_priors[1] + sum(w_theta_ham[c])
            # if spam return 1 else return 0
            result[c] = (1 if args_spam > args_ham else 0)

        return result



def create_classifier():
    classifier = SpamClassifier()
    classifier.train()
    return classifier

classifier = create_classifier()

if True:
    testing_spam = np.loadtxt(open("testing_spam.csv"), delimiter=",").astype(np.int)
    test_data = testing_spam[:, 1:]
    test_labels = testing_spam[:, 0]

    for classifier in [classifier1]:

        predictions = classifier.predict(test_data)
        #print(predictions)
        # print(test_labels)
        accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]
        print(f"Accuracy on test data is: {accuracy}")

        fp = 0
        fn = 0
        tp = 0
        tn = 0
        for i in range(len(predictions)):
            if predictions[i] != test_labels[i]:
                if predictions[i] == 1:
                    fp += 1
                else:
                    fn += 1
            else:
                if predictions[i] == 1:
                    tp += 1
                else:
                    tn += 1

        print(f"FPs on test data is: {fp}")
        print(f"FNs on test data is: {fn}")
        print(f"TPs on test data is: {tp}")
        print(f"TNs on test data is: {tn}")
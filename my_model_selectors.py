import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        #best score
        score = float('Inf')
        new_score=float('Inf')

        #best model
        model = None

        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:
                new_model = self.base_model(num_states)
                logL = new_model.score(self.X, self.lengths)

                #d = data points
                #f = features
                d, f = self.X.shape
                #parameters
                p = np.square(num_states)+2*num_states*f-1

                #BIC = -2 * logL + p * logN
                new_score = -2 * logL + p*np.log(d)

                if new_score<score:
                    score=new_score
                    model=new_model
            except:
                pass




        return model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        #best score
        score = float('-Inf')
        new_score=float('-Inf')

        #best model
        model = None

        other_words=[]
        l_words = []

        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:
                new_model = self.base_model(num_states)
                logL = new_model.score(self.X, self.lengths)

                for word in self.words:
                    if word != self.this_word:
                        other_words.append(self.hwords[word])

                for word in other_words:
                    l_words.append(new_model.score(word[0],word[1]))

                new_score = logL - np.mean(l_words)
                #scores.append(new_score)
                #models.append(new_model)


                if new_score > score:
                    score = new_score
                    model = new_model



            except:
                pass


        return model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        
        ## CALL:
        ## 
        """
        model = SelectorCV(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
        """
        """
        From documentation:
        K-Folds cross-validator

        Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).

        Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
        """

        #best score
        score = float('-Inf')
        new_score=float('-Inf')

        #best model
        model = None

        #splits -  3 by default

        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:
        
                logL = []
                if len(self.lengths)>2:
                    try:
                        seq = KFold(n_splits=3).split(self.sequences)
                        for train_set, test_set in seq:
                            self.X, self.lengths = combine_sequences(train_set, self.sequences)
                            test_x, test_lengths = combine_sequences(test_set, self.sequences)

                            new_model = self.base_model(num_states)
                            logL.append(new_model.score(test_x, test_lengths))

                        new_score = np.mean(logL)

                        
                    except:
                        pass
                else:
                    try:
                        new_model = self.base_model(num_states)
                        logL.append(new_model.score(self.X, self.lengths))
                        new_score = np.mean(logL) 
                    except:
                        pass
                #check if new score is best score
                        #update score and model if necesary
                if new_score>score:
                    score = new_score
                    model = new_model
            except :
                pass

        return model
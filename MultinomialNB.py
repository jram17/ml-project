import numpy as np
from collections import defaultdict

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_log_probs = {}
        self.vocab_size = 0
        self.classes = None
    
    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier from training data.
        """
        # Handle various input types
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.array(X)
        y = np.array(y)
        
        # Error handling for empty input
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data cannot be empty")
        
        # Compute unique classes and vocabulary size
        self.classes = np.unique(y)
        self.vocab_size = X.shape[1]
        
        # Initialize count tracking
        class_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: np.zeros(self.vocab_size))
        
        # Count feature occurrences per class
        for features, label in zip(X, y):
            class_counts[label] += 1
            feature_counts[label] += features
        
        total_samples = len(y)
        
        # Compute class priors (log probabilities)
        self.class_priors = {
            cls: np.log(class_counts[cls] / total_samples)
            for cls in self.classes
        }
        
        # Compute feature log probabilities with Laplace smoothing
        self.feature_log_probs = {}
        for cls in self.classes:
            # Add-one smoothing
            smoothed_feature_count = feature_counts[cls] + 1
            smoothed_total_count = smoothed_feature_count.sum()
            self.feature_log_probs[cls] = np.log(smoothed_feature_count / smoothed_total_count)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        # Handle sparse-like matrices by converting to dense NumPy array
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        
        # Error handling for empty input
        if len(X) == 0:
            raise ValueError("Input data cannot be empty")
        
        predictions = []
        
        for features in X:
            log_probs = {}
            for cls in self.classes:
                # Calculate log probability for each class
                log_prob = self.class_priors[cls] + np.dot(features, self.feature_log_probs[cls])
                log_probs[cls] = log_prob
            
            # Predict class with highest log probability
            predictions.append(max(log_probs, key=log_probs.get))
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        """
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        probabilities = []
        
        for features in X:
            log_probs = {}
            for cls in self.classes:
                log_prob = self.class_priors[cls] + np.dot(features, self.feature_log_probs[cls])
                log_probs[cls] = log_prob
            
            # Convert log probabilities to actual probabilities
            max_log_prob = max(log_probs.values())
            exp_probs = {cls: np.exp(prob - max_log_prob) for cls, prob in log_probs.items()}
            total = sum(exp_probs.values())
            class_probas = {cls: prob/total for cls, prob in exp_probs.items()}
            
            probabilities.append(class_probas)
        
        return probabilities
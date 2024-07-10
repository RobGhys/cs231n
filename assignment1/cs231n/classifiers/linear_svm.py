from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg, delta: int = 1):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        #print(f'shape X[i] -> {X[i].shape} | shape W -> {W.shape}')
        scores = X[i].dot(W)  # X[i] has shape (D, ) and W has shape (D, C) --> X[i] . scores has shape (C, )
        #print(f'scores shape: {scores.shape}')
        #print(f'scores: {scores}')
        #print(f'y [i]: {y[i]}')
        correct_class_score = scores[y[i]]
        #print(f'correct class score: {correct_class_score}')
        for j in range(num_classes):
            # the correct class is excluded from loss computation
            if j == y[i]:  # correct class
                continue
            margin = scores[j] - correct_class_score + delta  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]  # accumulate gradient for incorrect class
                dW[:, y[i]] -= X[i]  # accumulate gradient for correct class

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead, so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)  # shape:  (N, D) . (D, C) -> (N, C)
    # We take all the rows (X.shape[0] = N) and we create a range from 0 to N-1 (np.arange(...))
    # X.shape[0] has shape N. y has shape N
    # scores [..., y] -> we get for each int from 0 to N-1 the correct class (y). Shape (N,)
    # Then, we must reshape to (N,1) for broadcasting
    # (in the margins variable: both matrices must have compatible shapes)
    correct_class_scores = scores[np.arange(X.shape[0]), y].reshape(-1, 1)  # shape (N, 1)
    margins = np.maximum(0, scores - correct_class_scores + 1)  # shape (N, C)
    margins[np.arange(X.shape[0]), y] = 0  # correct class has 0's everywhere

    loss = np.sum(margins) / X.shape[0]  # averaged by N
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margin_mask = (margins > 0).astype(int)  # shape (N, C)
    margin_mask[np.arange(X.shape[0]), y] = - np.sum(margin_mask, axis=1)

    dW = X.T.dot(margin_mask) / X.shape[0]
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

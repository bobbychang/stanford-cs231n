import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  losses = np.zeros((num_train, num_classes))

  #print('Naive scores:')
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1  # note delta = 1
      if margin > 0:
        loss += margin
        losses[i,j] = margin
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
        #print("X[%d] added to col %d and subtracted from col %d" % (i, j, y[i]))
        #print("Naive gradient so far:")
        #print(dW)
    #print(scores)


  #print('Naive losses:')
  #print(losses)

  #print('Naive gradient before regularization:')
  #print(dW)


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  #print("Vectorized scores:")
  #print(scores)

  correct_class_scores = scores.reshape(-1)[np.array(range(num_train))*num_classes + y]

  #print("correct class scores:")
  #print(correct_class_scores)

  correct_class_matrix = np.ones((num_train, num_classes))
  correct_class_matrix.reshape(-1)[np.array(range(num_train))*num_classes + y] -= 1
  #print("correct_class_matrix:")
  #print(correct_class_matrix)


  losses = ((scores+1).T - correct_class_scores).T * correct_class_matrix
  #print("Vectorized losses (with adjustment)")
  #print(losses)

  losses = np.maximum(losses, np.zeros(losses.shape))
  #print("Vectorized losses (with adjustment + hinge):")
  #print(losses)

  loss = np.sum(losses)
  #print("Vectorized total loss:")
  #print(loss)

  #print("X.T")
  #print(X.T)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  losses[losses > 0] = 1
  grad_mult = losses + ((correct_class_matrix-1).T * np.sum(losses, axis=1)).T
  #print("Gradient multiplier matrix:")
  #print(grad_mult)

  dW = np.matmul(X.T, grad_mult)
  #print("Vectorized gradient matrix before regularization:")
  #print(dW)

  dW /= num_train
  dW += 2*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

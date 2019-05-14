import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  dim, num_classes = W.shape
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in np.arange(num_train):
    #print("x")
    #print(X[i])

    scores = X[i].dot(W)

    #print("scores")
    #print(scores)

    scores_exp_sum = 0
    for j in np.arange(num_classes):
      scores[j] = np.exp(scores[j])
      scores_exp_sum += scores[j]

    #print("scores after exponent")
    #print(scores)

    #print("scores_exp_sum")
    #print(scores_exp_sum)

    loss += -1 * np.log(scores[y[i]] / scores_exp_sum)

    #print("total loss so far")
    #print(loss)

    dw_from_x = np.zeros_like(dW)

    for d in np.arange(dim):
      for c in np.arange(num_classes):
        dw_from_x[d][c] = scores[c]
        if c == y[i]:
          dw_from_x[d][c] -= scores_exp_sum
        dw_from_x[d][c] *= X[i][d]
    #print("dw_from_x before coefficients")
    #print(dw_from_x)

    dw_from_x /= scores_exp_sum

    #print("dw_from_x")
    #print(dw_from_x)

    dW += dw_from_x

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  dim, num_classes = W.shape
  num_train = X.shape[0]

  scores = np.exp(X.dot(W))
  # print("scores after exponent")
  # print(scores)

  scores_exp_sum = np.sum(scores, axis=1)
  # print("scores_exp_sum")
  # print(scores_exp_sum)

  correct_class_matrix = np.zeros((num_train, num_classes))
  correct_class_matrix.reshape(-1)[np.arange(num_train) * num_classes + y] = 1

  loss = -1 * np.sum(np.log(np.sum(scores * correct_class_matrix, axis=1) / scores_exp_sum))
  # print("loss")
  # print(loss)

  dW = np.sum(X.reshape(num_train, dim, 1) / scores_exp_sum.reshape((num_train, 1, 1)) *
              (scores.reshape((num_train, 1, num_classes)) -
               scores_exp_sum.reshape((num_train, 1, 1)) * correct_class_matrix.reshape((num_train, 1, num_classes))),
              axis=0)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1 (Algorithm from TA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 - Convert y into a vector of 0,1
Y = eye(num_labels)(y,:);

% 2 - Perform the forward propagation
A1 = [ones(size(X,1), 1) X];

Z2 = A1 *  Theta1';
A2 = sigmoid(Z2);
A2 = [ones(size(A2,1), 1) A2];

Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

% 3 - Cost Function, non-regularized
%size(Y)
%size(A3)
costPerExample = sum( Y.*log(A3) + (1 - Y).*log(1 - A3));
J = -(1/m)*sum(costPerExample);

% Part 2 (Algorithm from TA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Regularized component of the cost

TT1 = Theta1(:, 2:end);
TT2 = Theta2(:, 2:end);

thetaOnePart = sum(sum(TT1.*TT1));
thetaTwoPart = sum(sum(TT2.*TT2));
regCost = (lambda/(2*m))*(thetaOnePart + thetaTwoPart);
% Regularized cost
J = J + regCost;

% Part 3 (Algorithm from TA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delta3 = A3 - Y;
delta2 = delta3 * TT2 .* sigmoidGradient(Z2);

%delta3 = delta3(:, 2:end);
DELTA2 = A2' * delta3;
DELTA1 = A1' * delta2;

Theta2_grad_unreg = (1/m)*DELTA2'
Theta1_grad_unreg = (1/m)*DELTA1'

% Step 4(9th) by TA for gradient regularization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note: TT1 and TT2 which is Theta1 and Theta2 without bias term
Theta1_grad_reg = (lambda/m)*TT1;
Theta2_grad_reg = (lambda/m)*TT2;

Theta1_grad_reg = [zeros(size(Theta1_grad_reg, 1), 1) , Theta1_grad_reg]
Theta2_grad_reg = [zeros(size(Theta2_grad_reg, 1), 1) , Theta2_grad_reg]

Theta1_grad = Theta1_grad_unreg + Theta1_grad_reg
Theta2_grad = Theta2_grad_unreg + Theta2_grad_reg

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

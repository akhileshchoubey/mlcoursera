function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% a1 = x; (add a0 for layer 1)
A1 = [ones(size(X,1), 1) X];

% z2 = Theta1 * a1; a2 = g(z2); (add a0 for layer 2)
Z2 = A1 *  Theta1';
%size(X)
%size(Theta1')
%size(Z2)
A2 = sigmoid(Z2);
A2 = [ones(size(A2,1), 1) A2];

% z3 = Theta2 * a2; a3 = g(z3) = h-theta-x
Z3 = A2 * Theta2';
%size(Z3)
A3 = sigmoid(Z3);

% take the max of h-theta-x to return the index as the value of image
[x ix] = max(A3, [], 2)
p = ix;

% =========================================================================

end

function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%for i = 1:m
	%disp("-------------");
	%The trick is to know that hx should have m x 1 matrix
	%hx = each row * each theta (which is passed as transpose)
	%hx = sigmoid(X(i,:)*theta);
	%J = J - (y(i)*log(hx)+(1-y(i))*log(1-hx));
	%J = J - (y(i)*log(sigmoid(X(i,:)*theta))+(1-y(i))*log(1-sigmoid(X(i,:)*theta)));
%end
%disp("-------------");
%J=J/m

J = (1/m)*sum(-y'*log(sigmoid(X*theta))-(1-y')*log(1-sigmoid(X*theta)));

grad = (1/m)*X'*(sigmoid(X*theta)-y);

J += (lambda/(2*m))*sum(theta(2:end).^2);

%grad += [0 (lambda*theta(2:size(theta),1)')/m];
%grad = (1/m)*X'*(sigmoid(X*theta)-y)
%grad(0) = 0 %(1/m)*sum(X(1,:)'*sigmoid(X(1,:)*theta - y(0)));

reg_theta=theta(2:end);
%reg_X=X(:,2:end);
%reg_y=y(2:end);
%reg_grad = (1/m)*reg_X'*(sigmoid(reg_X*reg_theta) - reg_y);

grad = grad + (lambda/m)*[0 ; reg_theta];

% =============================================================

grad = grad(:);

end

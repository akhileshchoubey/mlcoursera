function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
grad

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta

disp("Starting cost function.......");
for i = 1:m
	%disp("-------------");
	%The trick is to know that hx should have m x 1 matrix
	%hx = each row * each theta (which is passed as transpose)
	hx = sigmoid(X(i,:)*theta);
	J = J - (y(i)*log(hx)+(1-y(i))*log(1-hx));
	%J = J - (y(i)*log(sigmoid(X(i,:)*theta))+(1-y(i))*log(1-sigmoid(X(i,:)*theta)));
end
%disp("-------------");
J=J/m
%Above calculation oculd be further vectoriezed as below - 
%J = (1/m)*sum(-y'*log(sigmoid(X*theta))-(1-y')*log(1-sigmoid(X*theta)));

disp("Starting gradient part.........");
%m = # of samples
%for j = 1:size(theta)
%	disp("-------------");
%	gd = 0;
%	t = 0;
%	for i = 1:m
%		gd = sum(X(i,j)'*sigmoid(X(i,:)*theta - y(i)));
%		%gd = gd + sigmoid(X(i,:)*theta - y(i))*X(i,j)
%	end
%	grad(j) = gd/m;
%end
%grad

%Above calculation could be further vectorized as below - 
grad = (1/m)*X'*(sigmoid(X*theta)-y)
% the below one is an accepted solution as well ...
%grad = (1/m)*(sigmoid(X*theta)-y)'*X

% =============================================================
end

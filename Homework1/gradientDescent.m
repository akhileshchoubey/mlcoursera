function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
	
	disp(sprintf('=======%0.0f========',iter));
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	
	predictions = X * theta;
	differences = (predictions - y);
	temp0 = sum(differences);
	%disp(differences);
	
	differences_theta1 = differences.*X(:,2);
	%disp(differences_theta1);
	temp1 = sum(differences_theta1);
	%disp(differences_theta1)
	
	theta(1) = theta(1) - temp0 * (alpha / m);
	%disp(theta(1));
	theta(2) = theta(2) - temp1 * (alpha / m);
	%disp(theta(2));
	
	%theta = theta - [temp0 * (alpha / m); temp1 * (alpha / m)];
	
	%disp(theta);
	
	% Save the cost J in every iteration   
    J_history(iter) = computeCost(X, y, theta);
	
    % ============================================================
	
end

end

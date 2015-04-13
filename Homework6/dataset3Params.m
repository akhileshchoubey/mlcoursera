function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% m x 1 matrix is easy to iterate as opposed to 1 x m
c_options=[0.01 0.03 0.1 0.3 1 3 10 30];
sigma_options=[0.01 0.03 0.1 0.3 1 3 10 30];
error_matrix=ones(64,3);

%%%%%%% uncomment the below section to calculate the solution %%%%%%

%row=1;
%for sigma = sigma_options,
%	for C = c_options,
%		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%		predictions = svmPredict(model, Xval);
%		error=mean(double(predictions ~= yval));
%		error_matrix(row, 1) = error;
%		error_matrix(row, 2) = sigma;
%		error_matrix(row, 3) = C;
%		row += 1;
%	end
%end
%min and max function returns the row with mim or max 1st column values and 
%in case of a tie it proceeds to next column and so on
%error_matrix(:, 1)
%[val, index] = min(error_matrix(:,1))
%final_values = error_matrix(index(1,1), :)
%sigma = final_values(:, 2)
%C = final_values(:, 3)
% =========================================================================

end

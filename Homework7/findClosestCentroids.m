function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i=1:size(X,1)
	%initialize 0 distances for all centroids
	distance_vector = zeros(K, 1);
	%X(i, :)
	%calculate X(i) distance from each centroid(j)
	for j=1:K
		%calculate distance from each centroids
		%centroids(j)
		distance_vector(j) = sum((X(i, :) - centroids(j, :)).^2);
	end
	%if current distance is less than distance(j), assign j to idx(i) else continue
	%distance_vector
	[value, index] = min(distance_vector);
	idx(i) = index;
	disp('=================================')
end

% =============================================================

end


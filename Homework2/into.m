function [J] = into(theta, X)
	J = 0;
	for i = 1:length(X),
		J = J + theta(i) * X(i);
	end;
end;

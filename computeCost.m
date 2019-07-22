function J = computeCost(X, y, theta)
%computeCost Compute cost for linear regression with multiple variables

m = length(y); % number of training examples

 
J = 0;

h = X*theta;
error = zeros(m,1);

for i = 1:m
    
    error(i,:) = h(i,:) - y(i,:); 
end

sqrdError = sum(error.^2);

J = (1/(2.*m)).*sqrdError;

% =========================================================================

end

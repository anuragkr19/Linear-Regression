function [theta] = gradientDescent(X, y, theta, alpha, num_iters)
%gradientDescent Performs gradient descent to learn theta

m = length(y); % number of training examples

%computing transpose of input vector
XTran = X'

for iter = 1:num_iters
    h = X * theta
    error = h - y;
    theta_change = ((alpha).*(1/m)).*XTran*error;
    theta = theta - theta_change;
end

end

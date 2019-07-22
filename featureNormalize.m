function [X_norm, mu, sigma] = featureNormalize(X)
%featureNormalize Normalizes the features in X 

m = length(X);
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu = mean(X);
sigma = std(X);
mu_matrix = ones(m,1) * mu;
sigma_matrix = ones(m,1)*sigma;

for j = 1:m
    X_norm(j,:) = (X(j,:) - mu_matrix(j,:));
end

for k =1:m
    X_norm(k,:) = X_norm(k,:)./sigma_matrix(k,:);
end

end

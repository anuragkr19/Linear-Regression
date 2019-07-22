function  functionCall(X,y,order,X_test,y_test)
theta = randi([0, 1],order,1);

noofiterations = 1500;
alpha = 0.0003;

%Erro cost computed on training data
J = computeCost(X,y,theta);
[theta] = gradientDescent(X, y, theta, alpha, noofiterations);

fprintf('Cost Calculated for training data :%f',J);

fprintf('\n');

for k = 1:length(theta)
    fprintf('Theta %d is %f ',k,theta(k));
    fprintf(' ');
    
end

%prediction on test data
for n = 1:length(y_test)
    predict1 = X_test(n,:) * theta;
    fprintf('\n Predicted target output as %f and expected output as %f\n ',predict1,y_test(n,:));
end

%error computed on test data
j_new = computeCost(X_test,y_test,theta);
fprintf('\nCost Calculated for testing data :%f',j_new);


% Plot the linear fit
figure;
scatter(X_test(:,2),y_test);
hold on;
% keep previous plot visible
yhat = X_test*theta;
p1 = polyfit(X_test(:,2),yhat,1);
f1 = polyval(p1,X_test(:,2));
plot(X_test(:,2),f1,'*');
legend('Test Data', 'Linear regression');
hold off;

end
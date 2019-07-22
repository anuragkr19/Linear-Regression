%external reference used
%https://www.coursera.org/learn/machine-learning/supplement/aEN5G/gradient-descent-for-multiple-variables
%https://www.coursera.org/learn/machine-learning/programming/8f3qT/linear-regression

function main()
%loading data
data = load('trainingdata.txt');
testData = load('testdata.txt');
X = data(:, 1:2);
X_test = testData(:,1:2);
y_test = testData(:,3);
y = data(:, 3);
m = length(X);
%getting input from user
order = input("Enter polynomial order for Linear Regression.Please enter value from this set [1,2,3,4]\n"); 


if(order == 1)
    % for order first 
    X = data(:,1:2); 
    X = [ones(m, 1) X];
    X_test = testData(:,1:2);
    n = length(X_test);
    X_test = [ones(n, 1) X_test];
    functionCall(X,y,3,X_test,y_test)
elseif(order == 2)
    % for second order
    X2 = [data(:,1).*data(:,1),data(:,2).*data(:,2),data(:,1).*data(:,2)]; 
    X2 = [data(:,1:2) X2];  
    %normalizing training data
    [X2, mu, sigma] = featureNormalize(X2);
    X2 = [ones(m, 1) X2];
    %normalizing testdata
    X_test_2 = [testData(:,1).*testData(:,1),testData(:,2).*testData(:,2),testData(:,1).*testData(:,2)];
    X_test_2 = [testData(:,1:2) X_test_2];
    n = size(X_test_2,1);
    new_mu_matrix = ones(n,1) * mu;   
    new_sigma_matrix = ones(n,1)*sigma;
    
    for j = 1:length(X_test_2)
          X_test_2(j,:) = (X_test_2(j,:) - new_mu_matrix(j,:));
    end

    for k =1:length(X_test_2)
          X_test_2(k,:) =  X_test_2(k,:)./new_sigma_matrix(k,:);
    end
    
    X_test_2 = [ones(n, 1) X_test_2];
    functionCall(X2,y,6,X_test_2,y_test);
elseif(order == 3)
    %for third order
     X =  [data(:,1).*data(:,1).*data(:,1),data(:,2).*data(:,2).*data(:,2)]
     X =  [[data(:,1).*data(:,1).*data(:,2),data(:,2).*data(:,2).*data(:,1)] X]
     X =  [[data(:,1).*data(:,1),data(:,2).*data(:,2),data(:,1).*data(:,2)] X]; %x2
     X =  [data(:,1:2) X];
     %Normalizing training data
    [X, mu, sigma] = featureNormalize(X);
     X =  [ones(m, 1) X];
     
     X_test =  [testData(:,1).*testData(:,1).*testData(:,1),testData(:,2).*testData(:,2).*testData(:,2)]
     X_test =  [[testData(:,1).*testData(:,1).*testData(:,2),testData(:,2).*testData(:,2).*testData(:,1)] X_test]
     X_test =  [[testData(:,1).*testData(:,1),testData(:,2).*testData(:,2),testData(:,1).*testData(:,2)] X_test]; %x2
     X_test =  [testData(:,1:2) X_test];
     
     %Normalizing Test data
     
     n = size(X_test,1);
     new_mu_matrix = ones(n,1) * mu;   
     new_sigma_matrix = ones(n,1)*sigma;
    
     for j = 1:length(X_test)
          X_test(j,:) = (X_test(j,:) - new_mu_matrix(j,:));
     end

     for k =1:length(X_test)
          X_test(k,:) =  X_test(k,:)./new_sigma_matrix(k,:);
     end
     %end normalizing training data
     X_test = [ones(n, 1) X_test];
     
     functionCall(X,y,10,X_test,y_test);
elseif(order == 4)
    %for polynomial order 4
     X =  [data(:,1).*data(:,1).*data(:,1).*data(:,1),data(:,2).*data(:,2).*data(:,2).*data(:,2)]
     X =  [[data(:,1).*data(:,1).*data(:,1).*data(:,2),data(:,2).*data(:,2).*data(:,2).*data(:,1)] X]
     X =  [[data(:,1).*data(:,1).*data(:,1),data(:,2).*data(:,2).*data(:,2)] X];
     X =  [[data(:,1).*data(:,1).*data(:,2),data(:,2).*data(:,2).*data(:,1)] X];
     X =  [[data(:,1).*data(:,1),data(:,2).*data(:,2),data(:,1).*data(:,2)] X]; 
     X =  [data(:,1:2) X];

    [X, mu, sigma] = featureNormalize(X);
     X =  [ones(length(X), 1) X];
     
     X_test_4 =  [testData(:,1).*testData(:,1).*testData(:,1).*testData(:,1),testData(:,2).*testData(:,2).*testData(:,2).*testData(:,2)]
     X_test_4 =  [[testData(:,1).*testData(:,1).*testData(:,1).*testData(:,2),testData(:,2).*testData(:,2).*testData(:,2).*testData(:,1)]  X_test_4]
     X_test_4 =  [[testData(:,1).*testData(:,1).*testData(:,1),testData(:,2).*testData(:,2).*testData(:,2)]  X_test_4];
     X_test_4 =  [[testData(:,1).*testData(:,1).*testData(:,2),testData(:,2).*testData(:,2).*testData(:,1)]  X_test_4];
     X_test_4 =  [[testData(:,1).*testData(:,1),testData(:,2).*testData(:,2),testData(:,1).*testData(:,2)]  X_test_4]; 
     X_test_4 =  [testData(:,1:2)  X_test_4];

     n = size(X_test_4,1);
     X_test_4 =  [ones(n,1) X_test_4];

     functionCall(X,y,14, X_test_4,y_test);
end

end







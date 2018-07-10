clear
% importing, normalizing and splitting data
rng(20171115, 'twister');
if (exist('train_data', 'var') ~= 1)
    train_data = load('images_train.csv');
    train_label = load('labels_train.csv');
    test_data = load('images_test.csv');
    test_label = load('labels_test.csv');
end

n = size(train_data, 2);
c = 10;
train_size = 50000;
p = randperm(size(train_data, 1));

% converting labels to one hot encoding
y_train = zeros(c, size(train_label, 1));
y_train(train_label + 1 + c * (0:size(train_label, 1)-1)') = 1;
y_train = y_train';

y_test = zeros(c, size(test_label, 1));
y_test(test_label + 1 + c * (0:size(test_label, 1)-1)') = 1;
y_test = y_test';

X_train_all = train_data(p, :);
y_train_all = y_train(p, :);
X_train = X_train_all(1:train_size, :);
X_dev = X_train_all(train_size+1:end, :);
y_train = y_train_all(1:train_size, :);
y_dev = y_train_all(train_size+1:end, :);
X_test = test_data;

%normalize the data
avg = mean(mean(X_train));
s = std(reshape(X_train, [], 1));

X_train = (X_train - avg) / s;
X_dev = (X_dev - avg) / s;
X_test = (X_test - avg) / s;

% training the network (without regularization)
m = train_size;
h1 = 300; % 300 units in hidden layer

% initialize the parameters
W1 = randn(h1, n);
b1 = zeros(h1, 1);
W2 = randn(c, h1);
b2 = zeros(c, 1);

num_epoch = 30;
batch_size = 1000;
num_batch = m / batch_size;
learning_rate = 5;
lambda = 0;
train_loss = zeros(num_epoch, 1);
dev_loss = zeros(num_epoch, 1);
train_accuracy = zeros(num_epoch, 1);
dev_accuracy = zeros(num_epoch, 1);
output_train=zeros(size(y_train));
output_dev=zeros(size(y_dev));

% Your code here
for i=1:1:num_epoch
    i
    for j=0:1:num_batch-1
    
        [dW1, db1, dW2, db2] = backward_prop(X_train((j*batch_size)+1:((j+1)*batch_size),:), y_train((j*batch_size)+1:((j+1)*batch_size),:), W1, b1, W2, b2, lambda);
        W1=W1+5*dW1;
       W2=W2+5*dW2;
       b1=b1+5*db1;
       b2=b2+5*db2;
    end
    [h_output, prob, loss] = forward_prop(X_train, y_train, W1, b1, W2, b2, lambda);
    train_loss(i)=loss;
    [value,index2]=max(prob');
    [value,index_original]=max(y_train');
    train_accuracy(i)=nnz(~(index2-index_original))/50000;
    [h_output, prob, loss] = forward_prop(X_dev, y_dev, W1, b1, W2, b2, lambda);
    [value2,index2]=max(prob');
    [value2,index_original]=max(y_dev');
    dev_accuracy(i)=nnz(~(index2-index_original))/10000;
    dev_loss(i)=loss; 
    
    
    
end



% Your code end here
% plotting and displaying results
figure(1);
plot(1:num_epoch, train_loss, 'r', ...
    1:num_epoch, dev_loss, 'b');
legend('training set', 'dev set');
xlabel('epochs');
ylabel('loss');
figure(2);
plot(1:num_epoch, train_accuracy, 'r', ...
    1:num_epoch, dev_accuracy, 'b');
legend('training set', 'dev set');
xlabel('epochs');
ylabel('accuracy');

%[~, test_pred, ~] = forward_prop(X_test, y_test, W1, b1, W2, b2, lambda);
%[~, y_c_test_pred] = max(test_pred, [], 2);
%[~, y_c_test] = max(y_test, [], 2);
%test_accuracy = sum(y_c_test_pred == y_c_test) / size(y_test, 1);
%fprintf('test set accuracy: %f \n', test_accuracy);

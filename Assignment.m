A = load('breast-cancer-wisconsin.data');
B = A(:,[2:10]);
%benign    = find(A(:,11) == 4); 
%malignant = find(A(:,11) == 2);
trainInput = B([1:500],:);

testInput = B([501:699],:);
trainOutput = A([1:500],end);

testOutput = A([501:699],end);

net = newff(trainInput',trainOutput',3,{'tansig' 'tansig'},'trainr','learngd','mse');

net.trainPAram.goal = 0.01;
net.trainParam.epochs = 100;
net.trainParam.lr = 0.001;
net.trainParam.max_fail = 200;
net = train(net,trainInput',trainOutput');
output = net(testInput');
output(output>3)= 4;
output(output<3)= 2;

error = size(find(output~=testOutput'))
mse  = error/size(output)*100


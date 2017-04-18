classNum = 10;
fileName = 'data/train-images.idx3-ubyte';
[images, rowNum, columnNum, itemNum] = readImages(fileName);
% channel for RGB images %
channel = 1;
% reshape the images with channel,
% the minist is one channel images
images = reshape(images, rowNum, columnNum, channel, itemNum);
fileName = 'data/train-labels.idx1-ubyte';
labels = readLabels(fileName);
labels(labels == 0) = 10;

% kernel size and number %
kernel1Size = 5;
kernel1Num = 8;
% initialize weights and biases of the 1st convolutional layer %
kernel1 = random('norm', 0, 1, kernel1Size, kernel1Size, channel, kernel1Num) / sqrt(kernel1Size ^ 2);
bias1 = ones(1, kernel1Num) .* 0.01;
% initalize the velocity %
kernel1Vel = zeros(size(kernel1));
bias1Vel = zeros(size(bias1));

% stride for pooling scale %
stride = 2;
% initialize weights and biases of the 2nd convolutional layer %
kernel2Size = 5;
kernel2Num = 16;
kernel2 = random('norm', 0, 1, kernel2Size, kernel2Size, kernel1Num, kernel2Num) ./ sqrt(kernel2Size ^ 2);
bias2 = ones(1, kernel2Num) .* 0.01;
% initalize the velocity %
kernel2Vel = zeros(size(kernel2));
bias2Vel = zeros(size(bias2));

% initialize weights and biases of the Softmax layer %
kernelSSize = (((rowNum - kernel1Size + 1) / 2 - kernel2Size + 1) / 2) ^ 2 * kernel2Num;
% kernelS is 1 * kenelSSize vector %
kernelS = random('norm', 0, 1, classNum, kernelSSize) ./ sqrt(kernelSSize);
biasS = ones(classNum, 1) .* 0.01;
% initalize the velocity %
kernelSVel = zeros(size(kernelS));
biasSVel = zeros(size(biasS));

% update kernels and biases via momentem-based gradient
% init regularization parameter, momentum coefficient and learning rate
lambda = 0.001;
u = 0.5;
eta = 0.5;
speedUp = 20;
uMax = 0.9;
% process the image in batch = 100 %
batch = 200;
it = 0;
trainSet = 50000;

for epoch = 1: 5
    randIndex = randperm(itemNum);
    trainedImages = images(:, :, channel, randIndex(1: trainSet));
    trainedLabels = labels(randIndex(1: trainSet)');
    C = [];
    for i = 1: batch : trainSet
        it = it + 1;
        if (it == speedUp)
            u = uMax;
        end
        batchImg = trainedImages(:, :, :, i:i+batch-1);
        batchLab = trainedLabels(i: i+batch-1);
        % forward pass %
        
		% convolution 1 %
		% calculate the convolution %
		convFeatureMap1 = convLayer(batchImg, kernel1, bias1);
		
		% pool 1, down sampling scale by 2 %
		poolFeatureMap1 = poolLayer(convFeatureMap1, stride);
		
		% convolution 2 %
		% calculate the convolution %
		convFeatureMap2 = convLayer(poolFeatureMap1, kernel2, bias2);
		
		% pool 2 %
		poolFeatureMap2 = poolLayer(convFeatureMap2, stride);
		
		% feature vector %
		softIn = reshape(poolFeatureMap2, [], batch);
		% softmax output %
		softOut = exp(kernelS * softIn + repmat(biasS, [1, batch]));
		softSum = sum(softOut, 1);
		softOut = softOut ./ repmat(softSum, [classNum, 1]);
        %softOut = exp(bsxfun(@plus, kernelS * softIn, biasS));
        %softSum = sum(softOut, 1);
        %softOut = bsxfun(@times, softOut, 1 ./ softSum);
		
		% turn labels to matrix %
		yIndex = sub2ind(size(softOut), batchLab', 1:batch);
		y = zeros(classNum, batch);
		y(yIndex) = 1;
		
		% calculate the cost via cross-entropy  and weight decay %
		ceCost = sum( -sum(y .* log(softOut) + (1 - y) .* log(1 - softOut)) ) / batch;
		wCost = (lambda / (2 * batch)) * (sum(kernel1(:) .^ 2) + sum(kernel2(:) .^ 2) + sum(kernelS(:) .^ 2));
		cost = ceCost + wCost;
        fprintf('epoch:%d, batch:%d, cost is %f\n',epoch, i, cost);
        
        [~, prediction] = max(softOut, [], 1);
		
		% calculate delta and gradient for softmax %
		% the first term is the derivative of cost at output a (i.e. softOut)
		% and the derivative of softmax active function is the output(exp act func) 
		% deltaSoft = -(y ./ softOut - (1 - y) ./ (1 - softOut)) .* softOut;
        deltaSoft = -(y - softOut);
		gradBiasS = sum(deltaSoft, 2) ./ batch;  % the average gradient for images in a batch
		gradKernelS = deltaSoft * softIn' ./ batch;  % the average gradient for images in a batch
		
		% calculate delta of convolution layer from pooling layer %
		deltaUnpooled2 = calDeltaOfPooling(size(convFeatureMap2), stride, kernelS, deltaSoft, true);
		
		% calculate delta and gradient for convlution 2 %
		[gradKernel2, gradBias2, deltaConv2] = calGradOfConv(deltaUnpooled2, convFeatureMap2, poolFeatureMap1);
		
		% calculate delta of pooling layer1 %
		deltaUnpooled1 = calDeltaOfPooling(size(convFeatureMap1), stride, kernel2, deltaConv2, false);
		
		% calculate gradient of convolutional layer 1 %
		[gradKernel1, gradBias1, deltaConv1] = calGradOfConv(deltaUnpooled1, convFeatureMap1, batchImg);
		
		% kernel1 and bias1 of convolutional layer 1 %
		kernel1Vel = kernel1Vel * u - eta * (gradKernel1 + lambda / batch * kernel1);
		kernel1 = kernel1 + kernel1Vel;
		bias1Vel = bias1Vel * u - eta * gradBias1;
		bias1 = bias1 + bias1Vel;
		% kernel2 and bias2 of convolutional layer 2 %
		kernel2Vel = kernel2Vel * u - eta * (gradKernel2 + lambda / batch * kernel2);
		kernel2 = kernel2 + kernel2Vel;
		bias2Vel = bias2Vel * u - eta * gradBias2;
		bias2 = bias2 + bias2Vel;
		% kernelS and biasS of softmax layer %
		kernelSVel = kernelSVel * u - eta * (gradKernelS + lambda / batch * kernelS);
		kernelS = kernelS + kernelSVel;
		biasSVel = biasSVel * u - eta * gradBiasS;
		biasS = biasS + biasSVel;
        
        C(length(C) + 1) = cost;
    end
    
    eta = eta / 2;
    % test %
    fileName = 'data/t10k-images.idx3-ubyte';
    [testImages, testRowNum, testColumnNum, testItemNum] = readImages(fileName);
    % reshape the images with channel,
    % the minist is one channel images
    testImages = reshape(testImages, testRowNum, testColumnNum, channel, testItemNum);
    fileName = 'data/t10k-labels.idx1-ubyte';
    testLabels = readLabels(fileName);
    testLabels(testLabels == 0) = 10;
    
    % test the network %
    % convolution 1 %
	% calculate the convolution %
	convFeatureMap1 = convLayer(testImages, kernel1, bias1);		
	% pool 1, down sampling scale by 2 %
	poolFeatureMap1 = poolLayer(convFeatureMap1, stride);		
	% convolution 2 %
	% calculate the convolution %
	convFeatureMap2 = convLayer(poolFeatureMap1, kernel2, bias2);		
	% pool 2 %
	poolFeatureMap2 = poolLayer(convFeatureMap2, stride);
	% feature vector %
	softIn = reshape(poolFeatureMap2, [], testItemNum);
	% softmax output %
	softOut = exp(kernelS * softIn + repmat(biasS, [1, testItemNum]));
	softSum = sum(softOut, 1);
	softOut = softOut ./ repmat(softSum, [classNum, 1]);
    
    [~, prediction] = max(softOut, [], 1);
    prediction = prediction';
    acc = sum(prediction == testLabels) / testItemNum;
    fprintf('Accuracy is %f\n',acc);
    plot(C);
end
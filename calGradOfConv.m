function [gradKernel, gradBias, deltaConv] = calGradOfConv(deltaUnpooled, featureMap, input)
% this is for calculate delta of this convolutional layer and the gradient
% deltaUnpooled is from next layer after unpooling
% featureMap is the feature map of this convolutional layer
% input is the feature map of last layer(pooling or raw images layer)

% calculate delta for this convolutional layer %
% featureMap .* (1 - featureMap) is actually the f'(derivative of active function) %
deltaConv = deltaUnpooled .* featureMap .* (1 - featureMap);

% calculate gradient for convlution 2 %
% get size and number of objects %
mapLayerNum = size(input, 3);
imgNum = size(featureMap, 4);
kernNum = size(featureMap, 3);
kernSize = size(input, 1) - size(deltaConv, 1) + 1;
gradKernel = zeros(kernSize, kernSize, mapLayerNum, kernNum);
gradBias = zeros(1, kernNum);

for kernIndex = 1: kernNum
    for mapLayerIndex = 1: mapLayerNum
        for imgIndex = 1: imgNum
            deltaRot = rot90(deltaConv(:, :, kernIndex, imgIndex), 2);
            % we calculate the gradient by convolving the input map and the delta
            gradKernel(:, :, mapLayerIndex, kernIndex) = gradKernel(:, :, mapLayerIndex, kernIndex)...
                + conv2(input(:, :, mapLayerIndex, imgIndex), deltaRot, 'valid');
        end
    end
    temp = deltaConv(:, :, kernIndex, :);
    gradBias(kernIndex) = sum(temp(:));
end
% the average gradient for images in a batch
gradKernel = gradKernel ./ imgNum;
gradBias = gradBias ./ imgNum;
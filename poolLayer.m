function featureMap = poolLayer(input, stride)
% forward %
% pooling layer %
% input is feature map from last layer, and input this pooling layer %
% output is a feature map with down pooling %

mapSize = size(input, 1);
imgNum = size(input, 4);
% the input feature map is a cube because of 
% multiple convolved kernels of last convolved layer 
mapLayerNum = size(input, 3);
% this for mean down pooling, each element is 0.25 %
poKernel = ones(stride) ./ stride ^ 2;
featureMap = zeros(mapSize / stride, mapSize / stride, mapLayerNum, imgNum);
%convResult = zeros(mapSize - stride + 1);
for imgIndex = 1: imgNum
    for mapLayerIndex = 1: mapLayerNum
        convResult = conv2(input(:, :, mapLayerIndex, imgIndex), poKernel, 'valid');
        featureMap(:, :, mapLayerIndex, imgIndex) = convResult(1:stride:end, 1:stride:end);
    end
end
function featureMap = convLayer(input, W, B)
% forward pass %
% convolution %
% input is the input of convolved layer %
% W is weight matrix of kernal %
% B is a vector of bais %
% featureMap is output of this layer %

% get size and init %
kernelNum  = size(W, 4);
mapSize = size(input, 1);
convSize = mapSize - size(W, 1) + 1;
imgNum = size(input, 4);
% the input feature map is a cube because of 
% multiple convolved kernels of last convolved layer 
mapLayerNum = size(W, 3);
featureMap = zeros(convSize, convSize, kernelNum, imgNum);

for imgIndex = 1: imgNum
    for kerIndx = 1: kernelNum
        convResult = zeros(convSize);
        % this is for cube convolution, the 3rd dimension (or 'mapLayerIndex') 
        % comes from last convolved layer's output, 
        % which makes the input feature map of an image is an
        % imgSize * imgSize * mapLayerNum cube
        for mapLayerIndex = 1: mapLayerNum
            % rotate kernel with 180 %
            kernelRot = rot90( W(:, :, mapLayerIndex, kerIndx), 2 );
            % this is a cube convolution , kLast is the 3rd dimension of the kernel now we using %
            convResult = convResult + conv2(input(:, :, mapLayerIndex, imgIndex), kernelRot, 'valid');
        end
        % add bias %
        convResult = convResult + B(kerIndx);
        % active function is sigmoid %
        active = 1 ./ (1 + exp(-convResult));
        featureMap(:, :, kerIndx, imgIndex) = active;
    end
end
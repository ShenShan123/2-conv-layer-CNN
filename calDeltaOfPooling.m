function deltaUnpool = calDeltaOfPooling(mapSizeVec, stride, W, delta, fromSoft)
% this is for calculate the delta from pooling layer
% and the delta is for last convolutional layer
% mapSizeVect is the output of last convolutional layer,
% i.e. the input of this pooling layer before reshape 
% stride is the pooling sacle
% W is from next layer
% delta is from next layer
% fromSoft is a flag whether W and delta come from softmax layer

% get size and number
mapSize = mapSizeVec(1);
mapLayerNum = mapSizeVec(3);
imgNum = mapSizeVec(4);
% if the kernel and delta come from softmax layer %
if (fromSoft)
    % reshape the delta of this pooling layer and unpool it %
    deltaPool = reshape(W' * delta, mapSize / stride, mapSize / stride, mapLayerNum, imgNum);
else
    % if this pooling layer is between two convolutional layers,
    % calculate the delta of this layer via convolutional
    kernNum = size(W, 4);
    deltaPool = zeros(mapSize / stride, mapSize / stride, mapLayerNum, imgNum);
    for imgIndex = 1: imgNum
        for kernIndex = 1: kernNum;
            for mapLayerIndex = 1: mapLayerNum
                deltaPool(:, :, mapLayerIndex, imgIndex) = deltaPool(:, :, mapLayerIndex, imgIndex)...
                    + conv2(delta(:, :, kernIndex, imgIndex), W(:, :, mapLayerIndex, kernIndex), 'full');
            end
        end
    end
end

deltaUnpool = zeros(mapSizeVec);
for imgIndex = 1: imgNum
    for mapLayerIndex = 1: mapLayerNum
        % the kron function extends the deltaPool matix and mean the delta element %
        deltaUnpool(:, :, mapLayerIndex, imgIndex) = kron(deltaPool(:, :, mapLayerIndex, imgIndex), ones(stride)) ./ stride ^ 2;
    end
end
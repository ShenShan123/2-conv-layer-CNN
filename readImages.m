% read images from MNIST %
function [images, rows, columns, items] = readImages(fileName)

file = fopen(fileName,'rb'); % r for read,b for big endian %
assert(file ~= -1, ['Could not open ', fileName, '']);

magic = fread(file, 1, 'int32', 0, 'b');
assert(magic == 2051, 'Wrong train file!');

items = fread(file, 1, 'int32', 0, 'b');
rows = fread(file, 1, 'int32', 0, 'b');
columns = fread(file, 1, 'int32', 0, 'b');
images = fread(file, inf, 'uint8', 0, 'b');

% reshape the row and column %
images = reshape(images, columns, rows, items);
images = permute(images, [2, 1, 3]);
images = reshape(images, size(images, 1), size(images, 2), size(images, 3));
% to float %
images = double(images) ./ 255;

fclose(file);
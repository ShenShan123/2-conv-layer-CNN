% read label from MNIST %
function labels = readLabels(fileName)

file = fopen(fileName, 'rb'); % r for read,b for big endian %
assert(file ~= -1, ['Could not open ', fileName, '']);

magic = fread(file, 1, 'int32', 0, 'b');
assert(magic == 2049, 'Wrong train file!');

items = fread(file, 1, 'int32', 0, 'b');
labels = fread(file, inf, 'uint8', 0, 'b');
assert(size(labels,1) == items, 'Mismatch in label count');

fclose(file);
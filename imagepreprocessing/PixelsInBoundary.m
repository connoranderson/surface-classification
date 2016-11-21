function [ pixelSet ] = PixelsInBoundary( binImage, regionNum )
% Finds set of all pixels given by a boundary

result = bwconncomp(binImage);
IND = result.PixelIdxList(regionNum);
IND = IND{1};

x_len = size(binImage,1);
y_len = size(binImage,2);
[x,y] = ind2sub([x_len y_len],IND);

pixelSet = [x,y];

end


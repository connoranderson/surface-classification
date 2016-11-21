function [ binImage ] = filterNoise( binImage , threshold )
% Removes small clusters (smaller than threshold) from image binImage

areaMeasurements = regionprops(binImage, 'Area');
centroidLocations = regionprops(binImage, 'Centroid');
[allAreas ind] = sort([areaMeasurements.Area],'ascend');
centroidLocations_sorted = centroidLocations(ind);

num_regions = allAreas(allAreas < threshold);

while length(num_regions) > 0

areaMeasurements = regionprops(binImage, 'Area');
centroidLocations = regionprops(binImage, 'Centroid');
[allAreas ind] = sort([areaMeasurements.Area],'ascend');

num_regions = allAreas(allAreas < threshold);

for i=1:length(num_regions)
   centroid = uint32(centroidLocations(ind(i),:).Centroid);
   binImage(centroid(2),centroid(1)) = 0;  
   if centroid(2)-1 > 0
    binImage(centroid(2)-1,centroid(1)) = 0; 
   end
   if centroid(1)-1 >0
    binImage(centroid(2),centroid(1)-1) = 0; 
   end
   if centroid(2)+1 < size(binImage,2)
    binImage(centroid(2)+1,centroid(1)) = 0; 
   end
   if centroid(1)+1 < size(binImage,1)
  	binImage(centroid(2),centroid(1)+1) = 0; 
   end
end

if length(allAreas) == 0 % Our image is now empty
    binImage(500,500) = 1; % create pseudo point in center to prevent edge case
end

end


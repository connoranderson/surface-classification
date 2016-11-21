%% CS229 Image Preprocessing
% Code by Connor Anderson

% Navigates to data folder and goes through images one by one, subtracting
% the flash images with the raw image to find the number of potential flash
% regions. The maximum region area, along with the number of regions are
% output into the text file: 'circleParams.txt' for use in primary learning 
% algorithm.

close all;
clear all;

% Load distance data 
fileID = fopen('Data/distanceData.txt','r');

% Variables Logging Missing Images
num_missing = 0;
num_processed = 0;
num_onlyOne = 0;

% Clear output txt files
fileID2 = fopen('circleParams.txt','w');
fclose(fileID2);
fileID3 = fopen('matrixOutput.txt','w');
fclose(fileID3);



% Iterator Init
i = 0;

while true
    i = i+1;
    
    % Get next image filename 
    close all;
    tline = fgetl(fileID);
    if tline == -1
       break 
    end
    pat = '\s+';
    n = regexp(tline, pat, 'split');
    space = ' ';
    base_filename = strcat(n(1),{space},n(2));
    filename_flash = strcat(n(1),{space},n(2),{space},'wflash.jpg');
    filename_noflash = strcat(n(1),{space},n(2),{space},'noflash.jpg');
    filename_flash = strrep(filename_flash{1},':','_');
    filename_noflash = strrep(filename_noflash{1},':','_');
    class_string = n{6};
    class = str2num(class_string(length(class_string)));
    distance = str2num(n{5});
    
    % Enter Data Folder
    cd Data
    
    % Check if image exists in directory
    if exist(filename_flash, 'file') == 2 && exist(filename_noflash, 'file') == 2
        num_processed = num_processed + 1;
        I = imread(filename_flash);
        figure(1), imshow(I,[])
        B = imread(filename_noflash);
        figure(2), imshow(B,[])

        % Exit Data Folder
        cd ..

        % Image subtraction
        normI = im2double(I);
        normB = im2double(B);
        Ip_norm = imsubtract(I,B);
        figure(3), imshow(Ip_norm,[])
        
        % Ignore regions outside of bounding box
        Leftbound = 350;
        Rightbound = 800;
        TopBound = 330;
        BottomBound = 450;
        Ip_norm = Ip_norm(TopBound:BottomBound,Leftbound:Rightbound,:);
        
        % Image normalization (only normalize if not a dark image)
        PixIntensityThresh = 60;
        if (norm(double(squeeze(max(max(Ip_norm))))) > PixIntensityThresh)
            Ip_norm = double(Ip_norm)./max(double(max(double(Ip_norm(:,:)))))*255; 
        end        
        
        % Image normalization
        Ip_norm = uint8(round(Ip_norm));
        figure(4), imshow(Ip_norm,[])
        imwrite(Ip_norm,'normsubtract.jpg')

        % Find mean pixel and use it as threshold for black/white
        meanPixel = squeeze(mean(mean(Ip_norm)));
        thresh = mean(meanPixel);
        % Remove all pixels below this threshold
        Ip_norm(Ip_norm<thresh) = 0;
        % Convert to greyscale
        Ip_norm_grey = rgb2gray(Ip_norm);
        figure(5), imshow(Ip_norm_grey)
%         figure(6), imhist(Ip_norm_grey)
        % Threshold image to isolate brightest colors
        level = 0.6;
        % Convert to binary image
        fThresh = im2bw(Ip_norm_grey, level) ;
        
%         % Filter out noise smaller than threshold
        noiseThresh = 10;
        fThresh = filterNoise(fThresh,noiseThresh);        
        
        figure(7), imshow(fThresh);

        areaMeasurements = regionprops(fThresh, 'Area');
        [allAreas ind] = sort([areaMeasurements.Area],'descend');
        
        if length(areaMeasurements) > 0
            
            NumRegions(i) = length(allAreas);
            
            centroidLocations = regionprops(fThresh, 'Centroid');            
            center = centroidLocations(ind(1));
            center = [center.Centroid(2) center.Centroid(1)]; % Because centroid gets returned in Y,X

            pixelSet = PixelsInBoundary(fThresh,ind(1)); % Finds pixels in the biggest region
            avgPixel = [0;0;0];
            varPixel = 0;
            for k=1:size(pixelSet,1)
               avgPixel = avgPixel + double(squeeze(I(pixelSet(k,1), pixelSet(k,2),:)));   
            end
            avgPixel = avgPixel/size(pixelSet,1);
            for k=1:size(pixelSet,1)
               varPixel = varPixel + norm(avgPixel - double(squeeze(I(pixelSet(k,1), pixelSet(k,2),:))))^2;   
            end        
            varPixel = varPixel/size(pixelSet,1);
            gamma = 2.2;
            Y = .2126 * avgPixel(1)^gamma + .7152 * avgPixel(2)^gamma + .0722 * avgPixel(3)^gamma; 
            L(i) = 116 * Y ^ 1/3 - 16;
            N(i) = norm(avgPixel);
            V(i) = varPixel;

            iter = 1;
            distances = [];

            for k =1:size(fThresh,1)
                for l=1:size(fThresh,2)
                    if fThresh(k,l) == 1
                        distances(iter) = norm([k,l]-center)^2;
                        iter = iter + 1;
                    end
                end
            end

            avgVar = mean(distances);
        else
            L(i) = 0;
            N(i) = 0;
            V(i) = 0;
            allAreas = [0];
            NumRegions(i) = 0;
            avgVar = 1000; % Set to large number to distinguish from images with a concentrated point
            
        end


        %% Output data to text file

        MaxRegionArea(i) = allAreas(1);
        Classification(i) = class;
        Distance(i) = distance;
        AvgVar(i) = avgVar;
        
        if NumRegions(i)>1
            RegionChange(i) = allAreas(1)/allAreas(2);
        else
            RegionChange(i) = allAreas(1); % If no second region, just set change = max region
        end
        
        % Output to Readable TXT Format
        fileID2 = fopen('circleParams.txt','a+');
        fmt = '\n';
        line = strcat(base_filename,' NumRegions:',num2str(NumRegions(i)),' MaxRegionArea:',num2str(MaxRegionArea(i)),' Classification:',num2str(class),' Distance:',num2str(distance), ' AreaFrac:',num2str(RegionChange(i)), ' AvgVar:',num2str(AvgVar(i)))
        fprintf(fileID2,'%s\n',line{1});
        fclose(fileID2);
        
        % Output to Unreadable Matrix Format
        fileID3 = fopen('matrixOutput.txt','a+');
        fmt = '\n';
        line = strcat(num2str(NumRegions(i)),{space},num2str(MaxRegionArea(i)),{space},num2str(Distance(i)),{space},num2str(RegionChange(i)),{space},num2str(AvgVar(i)),{space},num2str(Classification(i)));
        fprintf(fileID3,'%s\n',line{1});
        fclose(fileID3);
        
    else
        if exist(filename_flash, 'file') == 2 || exist(filename_noflash, 'file') == 2
            num_onlyOne = num_onlyOne+1;
        end        
        num_missing = num_missing+1;
        % Exit Data Folder
        cd ..
        fprintf('Could not find an image file');
    end    
end


fprintf('Summary: \r\n Missing: %d \r\n Processed: %d \r\n MissingOne: %d \r\n',num_missing,num_processed,num_onlyOne);

scatter(MaxRegionArea,Classification)
set(gca,'xscale','log')
title('MaxRegionArea')

figure, scatter(NumRegions,Classification)
set(gca,'xscale','log')
title('NumRegions')

figure, scatter(Distance,Classification) % Sanity check that our data isn't biased
set(gca,'xscale','log')
title('Distance')

figure, scatter(RegionChange,Classification) % Sanity check that our data isn't biased
set(gca,'xscale','log')
title('FirstRegion/SecondRegion')

figure, scatter(AvgVar,Classification) % Sanity check that our data isn't biased
set(gca,'xscale','log')
title('AvgVar')

figure, scatter(L,Classification)
title('L vs Class')

figure, scatter(N,Classification)
title('N vs Class')

figure, scatter(V,Classification)
title('V vs Class')
% 
% %%
% %Training = [NumRegions' MaxRegionArea' Distance' AvgVar' RegionChange'];
% Training = [MaxRegionArea' RegionChange'];
% Group = [Classification'];
%  
% %svmtrain(Training,Group,'kernel_function','rbf')
% %%
% test_size = 20;
% xtrain = Training((1:size(Training,1)-test_size),:);
% xtest = Training((size(Training,1)-test_size:end),:);
% ytrain = Group(1:size(Group,1)-test_size);
% ytest =  Group(size(Group,1)-test_size:end);
% Model = svmtrain(xtrain,ytrain);
% Vals = svmclassify(Model,xtest);
% Sol = Vals-ytest
% Sol = abs(Sol(Sol~=0));
% sum(Sol)/length(Vals)



% ----------REFERENCE----------
%         figure, imshow(fThresh)
%         imwrite(fThresh,'topHat_grey.jpg')

%         [B,L,N,A] = bwboundaries(fThresh,'noholes');
%         figure, imshow(fThresh); hold on;
%         colors=['b' 'g' 'r' 'c' 'm' 'y'];
%         for k=1:length(B),
%           boundary = B{k};
%           cidx = mod(k,length(colors))+1;
%           plot(boundary(:,2), boundary(:,1),...
%                colors(cidx),'LineWidth',2);
% 
%           %randomize text position for better visibility
%           rndRow = ceil(length(boundary)/(mod(rand*k,7)+1));
%           col = boundary(rndRow,2); row = boundary(rndRow,1);
%           h = text(col+1, row-1, num2str(L(row,col)));
%           set(h,'Color',colors(cidx),'FontSize',14,'FontWeight','bold');
%         end



%trainImageCategoryClassifier
%https://www.mathworks.com/help/vision/ref/trainimagecategoryclassifier.html

%% Train, Evaluate, and Apply Image Category Classifier
%%
% Load two image categories.

% Copyright 2015 The MathWorks, Inc.

    setDir  = fullfile('C:\Users\Aaron\Desktop\CS229\Project\','Glass and Bark Categorized-20161208T064517Z','Glass and Bark Categorized');
    imgSets = imageSet(setDir, 'recursive');
%% 
% Separate the two sets into training and test data. Pick 30% of images from each set for the training data and the remainder  70% for the test data.
    [trainingSets, testSets] = partition(imgSets, 0.6, 'randomize'); 
%% 
% Create bag of visual words.
    bag = bagOfFeatures(trainingSets,'Verbose',true);
%% 
% Train a classifier with the training sets.
    categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);
%% 
% Evaluate the classifier using test images. Display the confusion matrix.
    confMatrix = evaluate(categoryClassifier, testSets)
%% 
% Find the average accuracy of the classification.
    mean(diag(confMatrix))
%% 
tic
% Apply the newly trained classifier to categorize new images.
    img = imread(fullfile(setDir, 'bark', '2016-11-12 05_55_12 noflash.jpg'));
    [labelIdx, score] = predict(categoryClassifier, img);
% Display the classification label.
    categoryClassifier.Labels(labelIdx)
toc
%% Aaron's new code to loop thru each photo again and figure out its classificaiton


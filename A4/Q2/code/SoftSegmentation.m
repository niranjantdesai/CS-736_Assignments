% Performs segmentation using Hidden MRF + GMM + EM (soft segmentation)


%% Initialize and load
clc;
clear;
close all;

load('../data/assignmentSegmentBrainGmmEmMrf.mat');

K = 3; % num of gaussians

%% Part a) Initialize MRF params

% Note: a 4-neighborhood system is used with potential function nonzero on
% 2-cliques

% Generating valid maps for neighbors

validMapLeft = circshift(imageMask,1,2);
validMapRight = circshift(imageMask,-1,2);
validMapTop = circshift(imageMask,1,1);
validMapBottom = circshift(imageMask,-1,1);

beta = 5;

priorFunction = @(candidate_label,current_labels) EvaluateLabelPriors(...
    candidate_label,current_labels,beta,validMapLeft,validMapRight,...
    validMapTop,validMapBottom,imageMask);

%% Part b) Label initialization

% Using k-means for label initialization
% Motivation is that is gives quick division of the values into 3 classes

validImage = imageData(logical(imageMask));
[idx,C] = kmeans(validImage,K);

labelMap = zeros(size(imageData));

labelMap(logical(imageMask)) = idx;

%% Part c) Gaussian params initialization

% Using label initialization to get means and variances

means_init = C; % kmeans centroids

sigmas_init = zeros(K,1);

for i=1:K
    clusterVals = validImage(idx==i);
    sigmas_init(i) = sqrt(sumsqr(clusterVals - means_init(i))/length(clusterVals));
end

%% Part d) Perform Segmentation
xInit = labelMap;
% xInit = zeros(size(imageData));

[x,means,sigmas,iters] = PerformSegmentation(xInit,imageData,means_init,...
    sigmas_init,20,imageMask,priorFunction);

%% Viewing results

set1 = zeros(size(imageData));
set2 = zeros(size(imageData));
set3 = zeros(size(imageData));

set1(x==1) = imageData(x==1);
set2(x==2) = imageData(x==2);
set3(x==3) = imageData(x==3);

suffix = num2str(now);
imwrite(set1,strcat('../results/',suffix,'set1.png'));
imwrite(set2,strcat('../results/',suffix,'set2.png'));
imwrite(set3,strcat('../results/',suffix,'set3.png'));

%% IMSHOWS

figure(1)
imshow(imageData);

figure(2)
imshow(set1);

figure(3)
imshow(set2);

figure(4);
imshow(set3);


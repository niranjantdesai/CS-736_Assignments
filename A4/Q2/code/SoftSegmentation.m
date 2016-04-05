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

beta1 = 1.8;
beta2 = 0;  % No MRF prior on labels

priorFunction1 = @(candidate_label,current_labels) EvaluateLabelPriors(...
    candidate_label,current_labels,beta1,validMapLeft,validMapRight,...
    validMapTop,validMapBottom,imageMask);

priorFunction2 = @(candidate_label,current_labels) EvaluateLabelPriors(...
    candidate_label,current_labels,beta2,validMapLeft,validMapRight,...
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
% We're using the initial class means obtained using kmeans and the initial
% standard deviation estimates are the average distances of any pixel value
% from the respective class means. This gives us a quick and good estimate
% of the optimal standard deviations

means_init = C; % kmeans centroids

sigmas_init = zeros(K,1);

for i=1:K
    clusterVals = validImage(idx==i);
    sigmas_init(i) = sqrt(sumsqr(clusterVals - means_init(i))/length(clusterVals));
end

%% Part d) Perform Segmentation
xInit = labelMap;
% xInit = zeros(size(imageData));

fprintf('*** Starting modified ICM with beta = %f ***\n',beta1);
[x1,means1,sigmas1,iters1] = PerformSegmentation(xInit,imageData,means_init,...
    sigmas_init,20,imageMask,priorFunction1);
fprintf('\n*** Starting modified ICM with beta = %f ***\n',beta2);
[x2,means2,sigmas2,iters2] = PerformSegmentation(xInit,imageData,means_init,...
    sigmas_init,20,imageMask,priorFunction2);

%% Viewing results

set11 = zeros(size(imageData));
set21 = zeros(size(imageData));
set31 = zeros(size(imageData));
set12 = zeros(size(imageData));
set22 = zeros(size(imageData));
set32 = zeros(size(imageData));

set11(x1==1) = imageData(x1==1);
set21(x1==2) = imageData(x1==2);
set31(x1==3) = imageData(x1==3);
set12(x2==1) = imageData(x2==1);
set22(x2==2) = imageData(x2==2);
set32(x2==3) = imageData(x2==3);

% suffix = num2str(now);
% imwrite(set11,strcat('../results/',suffix,'set11.png'));
% imwrite(set21,strcat('../results/',suffix,'set21.png'));
% imwrite(set31,strcat('../results/',suffix,'set31.png'));
% imwrite(set12,strcat('../results/',suffix,'set12.png'));
% imwrite(set22,strcat('../results/',suffix,'set22.png'));
% imwrite(set32,strcat('../results/',suffix,'set32.png'));

%% Show images and report optimal estimates

figure()
imagesc(xInit)
title('Initial estimate for the label image')

figure()
imshow(imageData);
title('Corrupted image')

figure()
imshow(set11);
title('Optimal class membership image estimate 1 for beta = 1.8')

figure()
imshow(set21);
title('Optimal class membership image estimate 2 for beta = 1.8')

figure();
imshow(set31);
title('Optimal class membership image estimate 3 for beta = 1.8')

figure()
imagesc(x1)
title('Optimal label image estimate for beta = 1.8')

figure()
imshow(set12);
title('Optimal class membership image estimate 1 for beta = 0')

figure()
imshow(set22);
title('Optimal class membership image estimate 2 for beta = 0')

figure();
imshow(set32);
title('Optimal class membership image estimate 3 for beta = 0')

figure()
imagesc(x2)
title('Optimal label image estimate for beta = 0')

fprintf('\nChosen value of beta = %f',beta1);
fprintf('\nThe optimal estimates for the class means are [%f %f %f] for beta = 1.8\n',means1(1),means1(2),means1(3));
% CS 736: Assignment 4
% Date: April 3, 2016
% Authors: Niranjan Thakurdesai, Ayush Baid

clc;
clear all;
close all;

%% Loading image
load('../data/assignmentSegmentBrain.mat');
imgSize = size(imageData);

% Show corrupted image
figure()
imshow(imageData)
title('Corrupted image')

%% Defining parameters and neighbourhood mask
K = 3;  % Number of segments
q = 4;    % Fuzziness parameter
cInit = [0;0.5;1];     % Initialize class means
uInit = (1/K)*ones(imgSize(1),imgSize(2),K);   % Memberships, initialized with a uniform distribution
bInit = ones(imgSize);   % Bias field; initially chosen to be a constant intensity image

% Create neighbourhood mask
windowSize = 25;    % 25 x 25 weight window
windowRadius = floor(windowSize/2);
sigma = 2;
w = fspecial('gaussian', windowSize, sigma);

% Algorithm parameters
maxIters = 7;
J = zeros(maxIters,1);  % Objective function across iterations


%% Modified FCM

for i=1:maxIters
   u = memberships( w,imageData,cInit,bInit,imageMask,K,windowRadius,q );     % Keeping class means and bias fixed, update memberships
   uInit = u;
   c = classMeans( uInit,imageData,w,bInit,imageMask,q,K);  % Keeping memberships, multipliers and bias fixed, update class means
   cInit = c;
   b = bias( w,imageData,uInit,cInit,imageMask,windowRadius,K,q );   % Keeping memberships, multipliers and class means fixed, update bias
   bInit = b;
   J(i) = objEval( imageData,imageMask,windowRadius,w,c,b,u,q,K );    % Evaluate objective function in the current iteration
   fprintf('Value of the objective function at iteration %d = %f',i,J(i));
end

%% Show required images
% Showing optimal class membership image estimates
figure()
imshow(u(:,:,1))
title('Optimal class membership image estimate 1')
figure()
imshow(u(:,:,2))
title('Optimal class membership image estimate 2')
figure()
imshow(u(:,:,3))
title('Optimal class membership image estimate 3')

% Showing optimal bias-field image estimate
figure()
imshow(b)
title('Showing optimal bias-field image estimate')

%% Construct bias-removed image
A = zeros(imgSize);
for i=1:imgSize(1)
   for j=1:imgSize(2)
      A(i,j) = sum(u(i,j,:).*c); 
   end
end
A = A.*imageMask;

% Show bias-removed image
figure()
imshow(A)
title('Bias-removed image')

%% Construct residual image
R = imageData - A.*b;

% Show residual image
figure()
imshow(R)
title('Residual image')

%% Report parameters and initial estimates
fprintf('q = %f',q);

% Show neighbourhood mask
figure()
imshow(w)
title('Neighbourhood mask')

% Show initial estimates for the membership values
figure()
imshow(uInit(:,:,1))
title('Initial class membership image estimate 1')
figure()
imshow(uInit(:,:,2))
title('Initial class membership image estimate 2')
figure()
imshow(uInit(:,:,3))
title('Initial class membership image estimate 3')

fprintf('The initial estimates for the class means are [%f %f %f]',cInit(1),cInit(2),cInit(3));
fprintf('The optimal estimates for the class means are [%f %f %f]',c(1),c(2),c(3));
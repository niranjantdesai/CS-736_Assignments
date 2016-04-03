% CS 736: Assignment 4
% Date: April 3, 2016
% Authors: Niranjan Thakurdesai, Ayush Baid

%% Loading image
load('../data/assignmentSegmentBrain.mat');
imgSize = size(imageData);

%% Defining parameters and neighbourhood mask
K = 3;  % Number of segments
q = 1.5;    % Fuzziness parameter
classMeans = zeros(K,1);
classMeans = [0;0.5;1];     % Initialize class means
memberships = (1/K)*ones(imgSize(1),imgSize(2),K);   % Memberships, initialized with a uniform distribution
bias = ones(imgSize);   % Bias field; initially chosen to be a constant intensity image
maxIters = 7;
J = zeros(maxIters,1);  % Objective function across iterations

% Create neighbourhood mask
windowSize = 25;    % 25 x 25 weight window
windowRadius = floor(windowSize/2);
sigma = 2;
w = fspecial('gaussian', windowSize, sigma);

%% Modified FCM

for i=1:maxIters
   u = memberships( w,imageData,classMeans,bias,imageMask,K,windowRadius );     % Keeping class means and bias fixed, update memberships
   classMeans = classMeans( u,imageData,w,b,imageMask,q);  % Keeping memberships, multipliers and bias fixed, update class means
   bias( w,imageData,memberships,classMeans,bias,imageMask,windowRadius )   % Keeping memberships, multipliers and class means fixed, update bias
end
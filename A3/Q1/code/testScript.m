% CS 736: Assignment 3
% 19th March 2016

%% Loading the data
clc;
clear;
close all;
load('../data/assignmentImageReconstructionPhantom.mat');
xInit = ifft2(imageKspaceData); % Initial solution in gradient descent
imageKspaceMask;
imageKspaceData = imageKspaceMask.*imageKspaceData;

noiselessNorm = sqrt(sumsqr(abs(imageNoiseless)));

%% Gradient descent
close all
g = @(x) QuadraticFunction(x);

alpha = 0.99999;
[x,logCost,iters] = GradientDescent(xInit,imageKspaceData,g,100,alpha,imageKspaceMask);

figure(1)
plot(logCost(1:iters));

    
figure(2)
imshow(abs(x));

%% RRMSE
initialRRMSE = sqrt(sumsqr(abs(imageNoiseless-xInit)))/noiselessNorm;
rrmse = sqrt(sumsqr(abs(imageNoiseless-x)))/noiselessNorm;

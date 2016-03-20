% CS 736: Assignment 3
% 19th March 2016

%% Loading the data
clc;
clear;
close all;
load('../data/assignmentImageReconstructionBrain.mat');
imageKspaceData_ = fftshift(imageKspaceData);
imageKspaceMask = logical(imageKspaceMask);
xInit = ifft2(imageKspaceData); % Initial solution in gradient descent

[val,grad] = GetLikelihoodTerm(xInit,imageKspaceData,imageKspaceMask);

close all
g = @(x) QuadraticFunction(x);

alpha = 0.5;
[x,logCost,iters] = GradientDescent(xInit,imageKspaceData,g,100,alpha,imageKspaceMask);

figure(1)
plot(logCost(1:iters));

    
figure(2)
imshow(abs(x));


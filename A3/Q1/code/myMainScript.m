% CS 736: Assignment 3
% 19th March 2016

%% Loading the data
clc;
clear all;
close all;
load('../data/assignmentImageReconstructionPhantom.mat');

noiselessNorm = sqrt(sumsqr(abs(imageNoiseless)));
xInit = ifft2(imageKspaceData); % Initial solution in gradient descent
figure()
imshow(imageNoiseless)
title('Noiseless');

%% Using quadratic function prior
% close all;
g = @(x) QuadraticFunction(x);
alphaRange1 = [0.99985];
rrmse1 = zeros(length(alphaRange1),1);

for i=1:length(alphaRange1)
    alpha = alphaRange1(i);
    [x,logCostArray,iter,stepSize] = GradientDescent(xInit,imageKspaceData,g,100,alpha,imageKspaceMask);
    rrmse1(i) = sqrt(sumsqr(abs(imageNoiseless)-abs(x)))/noiselessNorm;
%     figure()
%     imshow(x);
%     str = sprintf('alpha = %f',alpha);
%     title(str);
%     figure(1);
%     plot(logCostArray(1:iter));
%     title('Log cost function');
end

% figure()
% plot(rrmse1)
% title('rrmse for quadratic prior');

figure()
imshow(x)
title('quadratic')

%% Using Huber function prior
% close all;
alphaRange2 = [0.99981]; 
lambdaRange2 = [0.1];
rrmse2 = zeros(length(alphaRange2),length(lambdaRange2));

for i=1:length(alphaRange2)
    for j=1:length(lambdaRange2)
        g = @(x) HuberFunction(x,lambdaRange2(j));
        alpha = alphaRange2(i);

        [x,logCostArray,iters] = GradientDescent(xInit,imageKspaceData,g,100,alpha,imageKspaceMask);
        rrmse2(i,j) = sqrt(sumsqr(abs(imageNoiseless)-abs(x)))/noiselessNorm;
%         figure(1);
%         plot(logCostArray(1:iters));
%         title('Log cost function');
    end
end

% **Getting the optimum params**
[val2,index] = min(min(rrmse2));
lambda2 = lambdaRange2(index);
[~,index] = min(rrmse2(:,index));
alpha2 = alphaRange2(index);

% figure()
% plot(rrmse2)
% 
% % **Plotting RRMSE for huber prior**
% figure(2);
% surf(lambdaRange2,alphaRange2,rrmse2);
% plot(rrmse2);
% title('RRMSE plot for huber prior');
% xlabel('lambda');
% ylabel('alpha');

figure()
imshow(x)
title('huber')

%% Using g3() prior
% close all;
alphaRange3 = [0.99996];
lambdaRange3 = [0.9] ;
rrmse3 = zeros(length(alphaRange3),length(lambdaRange3));

for i=1:length(alphaRange3)
    for j=1:length(lambdaRange3)
        g = @(x) G3Function(x,lambdaRange3(j));
        alpha = alphaRange3(i);

        [x,logCostArray,iters] = GradientDescent(xInit,imageKspaceData,g,100,alpha,imageKspaceMask);
        rrmse3(i,j) = sqrt(sumsqr(abs(imageNoiseless)-abs(x)))/noiselessNorm;
%         figure(1);
%         plot(logCostArray(1:iters));
%         title('Log cost function');
    end
end

% figure()
% plot(rrmse3)

[val3,index] = min(min(rrmse3));
lambda3 = lambdaRange3(index);
[~,index] = min(rrmse3(:,index));
alpha3 = alphaRange3(index);

% **Plotting**
% figure(3);
% surf(lambdaRange3,alphaRange3,rrmse3);
% title('RRMSE plot for g3 prior');
% xlabel('lambda');
% ylabel('alpha');

figure()
imshow(x)
title('g3')
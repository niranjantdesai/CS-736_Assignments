% Denoising a phantom MRI image


%% Loading the data
load('../data/assignmentImageDenoisingPhantom.mat');

%% A) RRSME of given noisy image

noiselessNorm = sqrt(sumsqr(abs(imageNoiseless)));
initialRRMSE = sqrt(sumsqr(abs(imageNoiseless)-abs(imageNoisy)))/noiselessNorm;

%% B) 1: Using quadratic function prior
g = @(x) QuadraticFunction(x);
alphaRange1 = 0:0.05:1;
rrmse1 = zeros(length(alphaRange1),1);

for i=1:length(alphaRange1)
    alpha = alphaRange1(i);

    [x,logCostArray,iters] = GradientDescent(imageNoisy,imageNoisy,g,100,alpha);
    rrmse1(i) = sqrt(sumsqr(abs(imageNoiseless)-abs(x)))/noiselessNorm;
%     figure(1);
%     plot(logCostArray(1:iters));
%     title('Log cost function');
end

[val1,index] = min(rrmse1);
alpha1 = alphaRange1(index);

figure(1);
plot(alphaRange1,rrmse1);
title('RRMSE vs alpha plot for quadratic prior');

%% B) 2: Using huber function prior

alphaRange2 = 0:0.1:0.8;
lambdaRange = 0:0.05:0.6;

rrmse2 = zeros(length(alphaRange2),length(lambdaRange));

for i=1:length(alphaRange2)
    for j=1:length(lambdaRange)
        g = @(x) HuberFunction(x,lambdaRange(j));
        alpha = alphaRange1(i);

        [x,logCostArray,iters] = GradientDescent(imageNoisy,imageNoisy,g,100,alpha);
        rrmse2(i,j) = sqrt(sumsqr(abs(imageNoiseless)-abs(x)))/noiselessNorm;
    %     figure(1);
    %     plot(logCostArray(1:iters));
    %     title('Log cost function');
    end
end

[val2,index2] = min(rrmse2);
alpha1 = alphaRange2(index);

figure(2);
surf(lambdaRange,alphaRange2,rrmse2);
title('RRMSE plot for huber prior');
xlabel('lambda');
ylabel('alpha');

%% B) 2: Using g3()

alphaRange1 = 0:0.1:0.5;
lambdaRange = 0:0.02:0.5;

rrmse3 = zeros(length(alphaRange1),length(lambdaRange));

for i=1:length(alphaRange1)
    for j=1:length(lambdaRange)
        g = @(x) G3Function(x,lambdaRange(j));
        alpha = alphaRange1(i);

        [x,logCostArray,iters] = GradientDescent(imageNoisy,imageNoisy,g,100,alpha);
        rrmse3(i,j) = sqrt(sumsqr(abs(imageNoiseless)-abs(x)))/noiselessNorm;
    %     figure(1);
    %     plot(logCostArray(1:iters));
    %     title('Log cost function');
    end
end

figure(3);
surf(lambdaRange,alphaRange1,rrmse3);
title('RRMSE plot for g3 prior');
xlabel('lambda');
ylabel('alpha');


%% C) 




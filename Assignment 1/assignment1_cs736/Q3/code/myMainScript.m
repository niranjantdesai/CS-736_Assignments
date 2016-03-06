% Reconstruction from missing data

% Note: assumption - search only on integer values of theta

% Search performed using brute-force

clear;
clc;

%% Dataset 1

load('../data/CT_Chest.mat');

% Scaling input image between 0 and 1
maxVal = max(max(imageAC));
minVal = min(min(imageAC));

imageAC = (imageAC-minVal)/(maxVal-minVal);

theta = 0:150;
offset = 0:179;% integer offset considered

N = length(offset);

% empty array for rrmse
rrmse = zeros(length(offset),1);

% frobenious norm for the input image
inp_frob_norm = sqrt(sumsqr(imageAC));

for i=1:N
    % perform radon and inverse radon using missing data only
    rf = radon(imageAC,theta+offset(i));
    
    % inverse transform using Ram-Lak filter
    reconstructedImage = iradon(rf,theta+offset(i),'linear','Ram-Lak',1,size(imageAC,1));
    rrmse(i,1) = sqrt(sumsqr(reconstructedImage-imageAC))/inp_frob_norm;
end


figure(1);
plot(rrmse);
title('RRMSE (Dataset-1)');

[~,index] = min(rrmse);

disp('Best reconstruction for theta starting from: ');
disp(offset(index));

% Again performing the projection and reconstruction to obtain the best
% reconstruction


rf = radon(imageAC,theta+offset(index));
reconstructedImage = iradon(rf,theta+offset(index),'linear','Ram-Lak',1,size(imageAC,1));

% rescaling output between 0 and 1

maxVal = max(max(reconstructedImage));
minVal = min(min(reconstructedImage));
rescaledOutput = (reconstructedImage-minVal)/(maxVal-minVal);

% defining colormap
myNumOfColors = 200; 
myColorScale = repmat((0:1/myNumOfColors:1)',1,3);

figure(2);
imagesc(imageAC);
colormap(myColorScale);
title('Input Image - Dataset #1');

figure(3);
imagesc(reconstructedImage);
colormap(myColorScale);
title('Reconstructed Image - Dataset #1');



%% Dataset 2

load('../data/myPhantom.mat');

% Scaling input image between 0 and 1
maxVal = max(max(imageAC));
minVal = min(min(imageAC));

imageAC = (imageAC-minVal)/(maxVal-minVal);

theta = 0:150;
offset = 0:179;% integer offset considered

N = length(offset);

% empty array for rrmse
rrmse = zeros(length(offset),1);

% frobenious norm for the input image
inp_frob_norm = sqrt(sumsqr(imageAC));

for i=1:N
    % perform radon and inverse radon using missing data only
    rf = radon(imageAC,theta+offset(i));
    
    % inverse transform using Ram-Lak filter
    reconstructedImage = iradon(rf,theta+offset(i),'linear','Ram-Lak',1,size(imageAC,1));
    rrmse(i,1) = sqrt(sumsqr(reconstructedImage-imageAC))/inp_frob_norm;
end


figure(4);
plot(rrmse);
title('RRMSE (Dataset-2)');

[~,index] = min(rrmse);

disp('Best reconstruction for theta starting from: ');
disp(offset(index));

% Again performing the projection and reconstruction to obtain the best
% reconstruction


rf = radon(imageAC,theta+offset(index));
reconstructedImage = iradon(rf,theta+offset(index),'linear','Ram-Lak',1,size(imageAC,1));

% rescaling output between 0 and 1

maxVal = max(max(reconstructedImage));
minVal = min(min(reconstructedImage));
rescaledOutput = (reconstructedImage-minVal)/(maxVal-minVal);

% defining colormap
myNumOfColors = 200; 
myColorScale = repmat((0:1/myNumOfColors:1)',1,3);

figure(5);
imagesc(imageAC);
colormap(myColorScale);
title('Input Image - Dataset #2');

figure(6);
imagesc(reconstructedImage);
colormap(myColorScale);
title('Reconstructed Image - Dataset #2');


%% Observations

% The RRMSE plots are not perfectly convex, but trying gradient descent multiple
% times with random initialization may help to achieve results faster when we
% search for offsets beyond integers






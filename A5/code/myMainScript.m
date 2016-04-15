% Shape analysis
clc
clear all;
close;

%% Loading data
load('../data/assignmentShapeAnalysis.mat');
numPoints = size(pointSets,2);
numSets = size(pointSets,3);

%% Transforming each pointset to pre-shape space
centroids = sum(pointSets,2)/numPoints;
temp = repmat(centroids,1,32,1);

preshapePointSets = pointSets-temp;

l2_norms = sqrt(sum(sum(preshapePointSets.^2,2),1));
temp = repmat(l2_norms,2,32,1);

preshapePointSets = preshapePointSets./temp;

%% Checking point sets

% ps1 = pointSets(:,:,1);
% ps1_reshaped = preshapePointSets(:,:,1);
% 
% figure();
% scatter(ps1(1,:),ps1(2,:),'r+');
% hold on
% scatter(ps1_reshaped(1,:),ps1_reshaped(2,:),'b+');
% hold off

%% Plotting intial point sets
color_list=jet(numSets);

figure();
hold on;
title('Point Sets');
for i=1:numSets
   scatter(pointSets(1,:,i),pointSets(2,:,i),8,'MarkerFaceColor',color_list(i,:)); 
end
hold off

%% Mean shape calculation
% using a random point set as the mean initialization
initIndex = unidrnd(numSets,1);
mean = preshapePointSets(:,:,initIndex);
newMean = mean;

% params for iteration 
diffThreshold = 1e-6;
maxIters = 25;

iter = 1;
diff = 1e3;
while(diff>diffThreshold && iter<maxIters)
    mean = newMean;
    for i=1:numSets
        R = procrustes(mean,preshapePointSets(:,:,i));
        preshapePointSets(:,:,i) = R*preshapePointSets(:,:,i);
    end
    
    % Finding optimal mean shape within each iteration
    newMean = sum(preshapePointSets,3)/numSets;
    
    % normalizing mean to bring it into preshape space; centroid already at
    % origin
    l2_norm = sqrt(sumsqr(newMean));
    newMean = newMean./l2_norm;
    
    
    % calculate the difference between the means
    diff = sqrt(sumsqr(mean-newMean));
    % disp(diff);
    iter = iter+1;
end

figure()
scatter(mean(1,:),mean(2,:));
title('Final Mean shape');


%% Plotting aligned point sets

color_list=jet(numSets);

figure();
hold on;
% scatter(mean(1,:),mean(2,:),16,'MarkerFaceColor',[0 0 0]);
plot(mean(1,:),mean(2,:),'-x','LineWidth',4,'Marker','*');
for i=1:numSets
   scatter(preshapePointSets(1,:,i),preshapePointSets(2,:,i),...
       4,'MarkerFaceColor',color_list(i,:)); 
end
title('Aligned Point Sets');
hold off

%% Computing principal modes of variation
vectorizedPoints = zeros(size(pointSets,1)*size(pointSets,2),size(pointSets,3));

cov = 0;
% mean subtracted vectorized points
for i=1:numSets
    vectorizedPoints(:,i) = [preshapePointSets(1,:,i) preshapePointSets(2,:,i)]' - vectorizedMean;
    cov = cov + vectorizedPoints(:,i)*vectorizedPoints(:,i)';
end
cov = cov/numSets;


% Eigenvalue decomposition
[V,D] = eig(cov);

eigenvals = diag(D);

figure()
scatter(1:length(eigenvals),eigenvals);
title('Variances along each principal mode');






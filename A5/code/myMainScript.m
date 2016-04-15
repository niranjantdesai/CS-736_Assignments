% Shape analysis

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
% figure(1);
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
% using the first point set as a point
mean = preshapePointSets(:,:,1);

% params for iteration 
diffThreshold = 1e-4;
maxIters = 25;

iter = 1;
diff = 1e3;
while(diff>diffThreshold && iter<maxIters)
    for i=1:numSets
        R = align(mean,preshapePointSets(:,:,i));
        preshapePointSets(:,:,i) = R*preshapePointSets(:,:,i);
    end
    newMean = sum(preShapePointSets,3)/numSets;
    
    % normalizing mean to bring it into preshape space; centroid already at
    % origin
    l2_norm = sqrt(sumsqr(newMean));
    newMean = newMean./l2_norm;
    
    
    % calculate the difference between the means
    diff = sqrt(sumsqr(mean-newMean));
    iter = iter+1;
end




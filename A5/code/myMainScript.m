% Shape analysis

%% Loading data
load('../data/assignmentShapeAnalysis.mat');
numPoints = size(pointSets,2);

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

%% 




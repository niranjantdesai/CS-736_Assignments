function [x,means,sigmas,maxIters] = PerformSegmentation( x,y,means,sigmas,maxIters,validMap,priorFunction)
%PerformSegmentation Perform segmentation using iterative convergence mode
% Input arguments:
% x - initial labels
% y - oberved image
% means - initial means of Gaussians
% sigmas - intial s.d of Gaussians
% maxIters
% validMap - the map indicating the valid regions


mem = zeros(size(x,1),size(x,2),length(means));
memNew = zeros(size(x,1),size(x,2),length(means));

xNew = zeros(size(x));
posterior = zeros(size(x));

logPosteriorBefore = 0;
logPosteriorAfter = 0;

for i=1:maxIters
    % getting memberships
    mem = GetMemberships(y,means,sigmas,x,validMap,priorFunction);
    
    % generating the new label maps
    [posterior,xNew] = max(mem,[],3);
    xNew = xNew.*validMap;
    
    logPosteriorBefore = log(sum(sum(posterior)));
    fprintf('Iter %d: log posterior before = %f\n',i,logPosteriorBefore);
    
    % Gettinh posterior for the new label map
    memNew = GetMemberships(y,means,sigmas,xNew,validMap,priorFunction);
    [posterior,~] = max(memNew,[],3);
    
    logPosteriorAfter = log(sum(sum(posterior)));
    fprintf('Iter %d: log posterior after = %f\n',i,logPosteriorAfter);
    
    
    % Getting new params
    [means,sigmas] = GetGaussianParams(y,mem,validMap);
    
    diff = any(x~=xNew);
    if ~diff
        break;
    end 
    x = xNew;
end


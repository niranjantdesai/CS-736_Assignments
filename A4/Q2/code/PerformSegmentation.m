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

xNew = zeros(size(x));
posterior = zeros(size(x));

logPosteriorBefore = 0;
logPosteriorAfter = 0;

for i=1:maxIters
    % get posterior
    posterior = GetPosterior(x,y,means,sigmas,validMap,priorFunction);
    logPosteriorBefore = sum(sum(log(posterior(logical(validMap)))));
    fprintf('Iter %d: log posterior before = %f\n',i,logPosteriorBefore);
    
    % getting memberships
    mem = GetMemberships(y,means,sigmas,x,validMap,priorFunction);
    
    % generating the new label maps
    [~,xNew] = max(mem,[],3);
    xNew = xNew.*validMap;
    
    % get posterior after update
    posterior = GetPosterior(xNew,y,means,sigmas,validMap,priorFunction);
    logPosteriorAfter = sum(sum(log(posterior(logical(validMap)))));
    fprintf('Iter %d: log posterior after = %f\n',i,logPosteriorAfter);
    
    if logPosteriorAfter<logPosteriorBefore
        break;
    end
    
    % Getting new params
    [means,sigmas] = GetGaussianParams(y,mem,validMap);
    
    diff = any(x~=xNew);
    if ~diff
        break;
    end 
    x = xNew;
end


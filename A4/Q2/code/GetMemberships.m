function mem = GetMemberships( y,means,sigmas,x,validMap,priorFunction)
%GetMemberships Calculates the membership as per the E-step of soft segmentation
% Input arguments
% y - observed image
% means - the mean of gaussians of GMM
% vars - variance of gaussians of GMM
% x - the current labelling
% validMap - the map indicating valid portions of the image
% priorFunction - the function handle for MRF with beta and validMaps already
% initialized

K = length(means);

% calculate the likelihood and prior term for each class
likelihood = zeros(size(y,1),size(y,2),3);
prior = zeros(size(y,1),size(y,2),3);
for i=1:K
    likelihood(:,:,i) = ((1/(sigmas(i)*sqrt(2*pi)))*exp(-(y-means(i)).^2/(2*sigmas(i)^2))).*validMap;
    prior(:,:,i) = priorFunction(i,x);    
end

% normalizing prior term of each pixel
norms = sum(prior,3);
for i=1:size(y,1)
    for j=1:size(y,2)
        prior(i,j,:) = prior(i,j,:)/norms(i,j);
    end
end

mem = likelihood.*prior;


% normalizing the memberships 
norms = sum(mem,3);
for i=1:size(y,1)
    for j=1:size(y,2)
        mem(i,j,:) = mem(i,j,:)/norms(i,j);
    end
end

% Setting memberships of invalid regions to be zero
memberships = zeros(size(x,1),size(x,2));
for i=1:K
    memberships = mem(:,:,i);
    memberships(~logical(validMap)) = 0;
    mem(:,:,i) = memberships;
end



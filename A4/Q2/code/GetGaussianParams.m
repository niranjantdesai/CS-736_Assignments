function [means,sigmas] = GetGaussianParams( y,mem,validMap )
%GetGaussianParams Evaluates params of GMM using observed image and memberships
% Input arguments:
% y - oberserved image
% mem - memberships (0 for invalid pixels)

K = size(mem,3); % number of gaussians

means = zeros(1,K);
sigmas = zeros(1,K);

for i=1:K
    den = sum(sum(mem(:,:,i)));
    means(i) = sum(sum(mem(:,:,i).*y))/den;
    sigmas(i) = sqrt(sum(sum(mem(:,:,i).*((y-means(i)).^2).*validMap))/den);
end


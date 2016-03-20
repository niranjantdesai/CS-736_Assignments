function [val,grad] = GetLikelihoodTerm_(x,y,mask)
%GetLikelihoodTerm terms the total likehood value to be used in the cost 
% function. Also returns the gradient at each point.

% Complex Gaussian is used to model noise
% Noise in real parts considered independent to that in imaginary parts. Hence,
% an identity covariance matrix is used

% Note: X and Y are vectorized

t1 = fft2(x);
t2 = (t1-y).*mask;

grad = 2*fft2(t2);
val = sum(sum(abs(t2).^2));



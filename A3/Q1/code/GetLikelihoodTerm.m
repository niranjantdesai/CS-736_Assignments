function [val,grad] = GetLikelihoodTerm(x,y,S)
%GetLikelihoodTerm terms the total likehood value to be used in the cost 
% function. Also returns the gradient at each point.

% Complex Gaussian is used to model noise
% Noise in real parts considered independent to that in imaginary parts. Hence,
% an identity covariance matrix is used

grad = 2*(ifft2(S.*fft2(x)) - ifft2(y));
val = sum(sum(abs(y-S.*fft2(x)).^2));

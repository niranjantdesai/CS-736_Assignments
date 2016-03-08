function [ val,grad ] = HuberFunction(u,lambda)
%HuberFunction Sum of the value of the huber function and its gradient at each
%point

uAbs = abs(u);

thresholdCriteria = uAbs<=lambda;

val = zeros(size(u));
grad = zeros(size(u));

val(thresholdCriteria)=0.5*uAbs(thresholdCriteria);
val(~thresholdCriteria)=lambda*uAbs(~thresholdCriteria)-0.5*(lambda^2);

grad(thresholdCriteria) = u(thresholdCriteria);
grad(~thresholdCriteria) = lambda*sign(u(~thresholdCriteria));

val = sum(sum(val));
end


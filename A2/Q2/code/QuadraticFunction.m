function [ val,grad ] = QuadraticFunction(u)
%QuadraticFunction Evaluation of the sum of function and its gradient at data u

uAbs = abs(u);
val = sum(sum(uAbs.^2));
grad = 2*u;

end


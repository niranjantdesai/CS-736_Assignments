function [ val,grad ] = G3Function(u,lambda)
%G3Function sum total of the evaluation of the function(g3) and its 
% gradient at  each datapoint u

uAbs = abs(u);

val = sum(sum(lambda*uAbs - (lambda^2)*log(1+uAbs/lambda)));
grad = u.*(lambda./(lambda+uAbs));

end


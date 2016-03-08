function [val,grad] = MRFEval(x,g)
%MRFEval Evaluates the contribution of the prior in the cost function for
%   each x(i). Also returns the contribution in the gradient

% 4-Neighbourhood with non-zero potential functions on 2 cliques
% g(u) = penalty function

% Creating the neighborhood difference arrays for x
% e.g. topArray(i) = x(i) - {the top neighbor for x(i)}

topArray = x-circshift(x,1,1);
bottomArray = x-circshift(x,-1,1);
leftArray = x-circshift(x,1,2);
rightArray = x-circshift(x,-1,2);

% val(i) = g(difference from top) + g(difference from bottom) + g(diff from
%       left) + g(diff from right)

[topVal,topGrad] = g(topArray);
[bottomVal,bottomGrad] = g(bottomArray);
[leftVal,leftGrad] = g(leftArray);
[rightVal,rightGrad] = g(rightArray);

val = topVal+bottomVal+leftVal+rightVal;
grad = topGrad+bottomGrad+leftGrad+rightGrad;

end


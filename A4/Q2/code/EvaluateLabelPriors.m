function p = EvaluateLabelPriors( candidate_label,x,beta,validMapLeft,...
    validMapRight,validMapTop,validMapBottom,validMap)
%EvaluateLabelPriors Evaluates prior on labels
%   Input arguments
%   candidate_label - the candidate for the whole image (evaluated at once for
%   speedup)
%   x - labels
%   beta - penalty for dissimilar labels
%   validMap - valid pixels

% Evaluating on 4-neighborhood system


if length(candidate_label)==1
    candidate_img = candidate_label*ones(size(x));
else
    candidate_img = candidate_label;
end

topArray = ((candidate_img-circshift(x,1,1)).*validMapTop)~=0;
bottomArray = ((candidate_img-circshift(x,-1,1)).*validMapBottom)~=0;
leftArray = ((candidate_img-circshift(x,1,2)).*validMapLeft)~=0;
rightArray = ((candidate_img-circshift(x,-1,2)).*validMapRight)~=0;


penalty = (topArray+bottomArray+leftArray+rightArray).*beta;

p  = exp(-penalty).*validMap;

end


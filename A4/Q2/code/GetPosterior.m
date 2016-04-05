function posterior = GetPosterior( x,y, means,sigmas, validMap, priorFunction )
%GetPosterior Summary of this function goes here
%   Detailed explanation goes here

idx1 = find(x==1);
idx2 = find(x==2);
idx3 = find(x==3);

likelihood = zeros(size(x));

likelihood(idx1) = (1/(sigmas(1)*sqrt(2*pi)))*exp(-(y(idx1)-means(1)).^2/(2*(sigmas(1)^2)));
likelihood(idx2) = (1/(sigmas(2)*sqrt(2*pi)))*exp(-(y(idx2)-means(2)).^2/(2*(sigmas(2)^2)));
likelihood(idx3) = (1/(sigmas(3)*sqrt(2*pi)))*exp(-(y(idx3)-means(3)).^2/(2*(sigmas(3)^2)));

prior = priorFunction(x,x);

posterior = likelihood.*prior.*validMap;



end


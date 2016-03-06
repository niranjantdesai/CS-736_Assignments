function [ J ] = obj( S,b,G,S0,L )
%obj Computes the objective function
%   J -> Value of the objective function at the given L

J = 0;
for i=1:size(S,2)
    c2 = exp(-b*G(i,:)*L*L'*G(i,:)');
    c1 = S(i) - S0*c2;
    J = J + c1^2;
end

end


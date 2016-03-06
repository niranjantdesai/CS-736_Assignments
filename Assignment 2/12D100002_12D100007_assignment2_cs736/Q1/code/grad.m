function [ J11,J21,J22 ] = grad( S,b,G,S0,L )
%obj Computes derivatives of the objective function wrt L
%   J -> Value of the objective function at the given L
%   J11 -> First derivative of the objective function wrt L11
%   J21 -> First derivative of the objective function wrt L21
%   J22 -> First derivative of the objective function wrt L22

J11 = 0;
J21 = 0;
J22 = 0;
for i=1:size(S,2)
    c2 = exp(-b*G(i,:)*L*L'*G(i,:)');
    c1 = S(i) - S0*c2;
    J11 = J11 + c1*c2*(G(i,1)^2*L(1,1) + G(i,1)*G(i,2)*L(2,1));
    J21 = J21 + c1*c2*(G(i,1)*G(i,2)*L(1,1) + G(i,2)^2*L(2,1));
    J22 = J22 + c1*c2*G(i,2)^2*L(2,2);
end

J11 = J11*8*b*S0;
J21 = J21*8*b*S0;
J22 = J22*8*b*S0;

end
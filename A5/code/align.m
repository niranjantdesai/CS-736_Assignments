function [ R ] = align( mean, test )
%align Gives the rotation between the mean and the test pointset
% x = test, y = mean

M = test*mean';
[U,~,V] = svd(M);
R = V*U';
if(det(R)==-1)
   t = [1,0;0,-1];
   R = V*t*U';
end

end


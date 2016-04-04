function [ b ] = bias( w,y,u,c,K,q )
%bias Finds the optimal value of the bias field within every iteration

imgSize = size(y);
b = zeros(imgSize);
innerSumNum = zeros(imgSize);
innerSumDenom = zeros(imgSize);

for k=1:K
   innerSumNum = innerSumNum + (u(:,:,k).^q)*c(k); 
   innerSumDenom = innerSumDenom + (u(:,:,k).^q)*(c(k)^2); 
end

num = conv2(y.*innerSumNum,w,'same');
denom = conv2(innerSumDenom,w,'same');

b = num./denom;

end


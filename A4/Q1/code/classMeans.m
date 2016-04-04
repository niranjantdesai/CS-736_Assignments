    function [ c ] = classMeans( u,y,w,b,mask,q,K )
%classMeans Finds the optimal value of class means within every iteration

imgSize = size(y);
c = zeros(K,1);
y = y.*mask;
innerSumNum = conv2(b,w);
innerSumDenom = conv2(b.^2,w);

for k = 1:K
    num = sum(sum((u(:,:,k).^q).*y.*innerSumNum));
    denom = sum(sum((u(:,:,k).^q).*innerSumDenom));
    c(k) = num/denom;
end

end


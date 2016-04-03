function [ c ] = classMeans( u,y,w,b,mask,q, )
%classMeans Finds the optimal value of class means within every iteration

imgSize = size(y);
c = zeros(K,1);
innerSumNum = conv2(b,w);
innerSumDenom = conv2(b.^2,w);
for k = 1:K
    num=0;
    denom=0;
    for i = 1:imgSize(1)
        for j = 1:imgSize(2)
            if (mask(i,j) > 0)
                num = num + innerSumNum(i,j)*(u(i,j,k)^q)*y(i,j);
                denom = denom + innerSumDenom(i,j)*(u(i,j,k)^q);
            end
        end
    end
    c(k) = num/denom;
end


end


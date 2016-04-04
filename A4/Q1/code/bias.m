function [ b ] = bias( w,y,u,c,mask,windowRadius,K,q )
%bias Finds the optimal value of the bias field within every iteration

imgSize = size(y);
b = zeros(imgSize);

for i = 1:imgSize(1)
    for j = 1:imgSize(2)
        if (mask(i,j) > 0)
            num1 = 0;
            denom1 = 0;
            for l = -windowRadius:windowRadius
                for m = -windowRadius:windowRadius
                    if(mask(i+l,j+m) > 0)
                        num2 = 0;
                        denom2 = 0;
                        for k = 1:K
                            num2 = num2 + (u(i+l,j+m,k)^q)*c(k);
                            denom2 = denom2 + (u(i+l,j+m,k)^q)*(c(k)^2);
                        end
                        num1 = num1 + w(l+windowRadius+1, m+windowRadius+1)*y(i+l,j+m)*num2;
                        denom1 = denom1 + w(l+windowRadius+1, m+windowRadius+1)*denom2;
                     end
                 end
             end
             b(i,j) = num1/denom1;
         end
    end
end

end


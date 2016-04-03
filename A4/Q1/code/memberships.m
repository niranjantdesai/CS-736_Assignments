function [ u ] = memberships( w,y,c,b,mask,K,windowRadius )
%memberships Finds optimal values of memberships within every iteration

imgSize = size(y);
% u = zeros(imgSize);
d = zeros(K,1);

for i=1:imgSize(1)
   for j=1:imgSize(2)
      if(mask(i,j)>0)
          for k = 1:K     
            for l = -windowRadius:windowRadius
                for m =  -windowRadius:windowRadius
                    d(k) = d(k) + w(l+windowRadius+1, m+windowRadius+1)*(y(i,j) - c(k)*b(i+l,j+m))^2;
                end
            end
          end
          u(i,j,:) = (1./d).^(1/(q-1));
          u(i,j,:) = u(i,j,:)/(sum(u(i,j,:)));
      end
   end
end

end


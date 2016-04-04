function [ u ] = memberships( w,y,c,b,mask,K,windowRadius,q )
%memberships Finds optimal values of memberships within every iteration

r = windowRadius;

imgSize = size(y);
u = zeros(imgSize(1),imgSize(2),K);
d = zeros(K,1);

for i=1:imgSize(1)
   for j=1:imgSize(2)
      if(mask(i,j)>0)
          for k = 1:K     
              d(k) = sum(w(0:2*r+1, 0:2*r+1)*((y(i,j) - c(k).*b(i+-r:r,j+-r:r)).^2));
          end
          u(i,j,:) = (1./d).^(1/(q-1));
          u(i,j,:) = u(i,j,:)/(sum(u(i,j,:)));
      end
   end
end

end


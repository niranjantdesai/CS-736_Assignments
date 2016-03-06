function  result = myIntegration( img,t,theta ,delS)
%myIntegration Performs integration on the image intensities along the line
%L(t,theta)


% x = t*cos(theta) - s*sin(theta) ... (a)
% y = t*sin(theta) + s*cos(theta) ... (b)

xMin = -floor((size(img,1)-1)/2);
xMax = floor(size(img,1)/2);

yMin = -floor((size(img,2)-1)/2);
yMax = floor(size(img,2)/2);

% Calculating the range of s
temp = [(t*cos(theta)-xMin)/sin(theta),(-t*sin(theta)+yMin)/cos(theta)];
[~,index] = min(abs(temp));
s1 = temp(index);

temp = [(t*cos(theta)-xMax)/sin(theta),(-t*sin(theta)+yMax)/cos(theta)];
[~,index] = min(abs(temp));
s2 = temp(index);

sMin = min(s1,s2);
sMax = max(s1,s2);

% Using derivatives of (a) and (b) to calculate step size as the min from that
% obtained by the two equations; del(x)=del(y)=1

if ~isempty(delS) && delS>0
    stepSize=delS;
else
    stepSize = min(abs([sin(theta),cos(theta)]));
end
            
s = sMin:stepSize:sMax;

% offseting the query points so that grid starts from (1,1)
x = t*cos(theta)-s*sin(theta)-xMin+1;
y = t*sin(theta)+s*cos(theta)-yMin+1;

% deleting gridpoints which will be outside the image

interpVals = interp2(img,x,y,'linear');

result = nansum(interpVals).*stepSize;
end


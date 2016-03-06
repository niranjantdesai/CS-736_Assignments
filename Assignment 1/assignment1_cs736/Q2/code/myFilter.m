function [ram_lak,shepp_logan,cosine] = myFilter( L )
%myFilter implements the Ram-Lak, Shepp-Logan and Cosine filters

w = (-L:L)/L;
ram_lak = abs(w);    % Ram-Lak filter
sinc = sin(0.5*w*pi/L)./(0.5*w*pi/L);
sinc(L+1)=1;
shepp_logan = sinc.*ram_lak;    % Shepp-Logan filter
cosine = cos(0.5*pi*w/L).*ram_lak;
end


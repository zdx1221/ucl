function K= kernel2(s,t,xi2)
if nargin<3
    xi2=0;
end
theta2=2;
tao=1;
sigma2=2;
fai2=0.25;
eta2=10;
delta2=4500;
lamda2=4489;
[~,a]=size(s);
[~,b]=size(t);
K=zeros(a,b);
for i=1:a
    for j=1:b
        K(i,j)=theta2*(exp((-2*sin(pi*(s(:,i)-t(:,j))/tao).^2)/sigma2)+ fai2*exp(-0.5*(s(:,i)-t(:,j)).^2/eta2))+delta2*exp(-0.5*(s(:,i)-t(:,j)).^2/lamda2)+xi2*all(s(:,i)'==t(:,j)');
    end
end
end
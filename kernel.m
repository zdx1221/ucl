function K = kernel(s, t, model, xi)

sigma = model.kernel_parameter{1};
A = model.kernel_parameter{2};

a = size(s,1);
b = size(t,1);
s = s.*repmat(A, a, 1);
t = t.*repmat(A, b, 1);
ss = sum(s.*s,2);
tt = sum(t.*t,2);
norm = repmat(ss,1,b)+repmat(tt',a,1)- 2*s*t';
K = sigma^2*exp(-0.5*norm);% + theta2*exp((-2*sin(pi*norm/tao).^2)/sigma2);

if nargin>4
    K=K+xi^2*eye(a);
end
end


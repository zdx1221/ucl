
function fx = eval_do_x(model, dat, x)

% eval_do_x:
%
% Evaluate E[Y | do(X)] for a given set of x values.
%
% Input:
%
% - x: the x_values
%
% Output:
%
% - fx: evaluation of do(X) at each of the given points.

if nargin == 1, x = (0:0.01:1)'; end

x_basis = [x sin(3*pi*x) sin(5*pi*x)];
fx = sum(x_basis .* repmat(model.fX', length(x), 1), 2) + trace(model.fZ{2}*dat.cov_Z);

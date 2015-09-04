
function dat = generate_data(model, N)

% generate_data:
%
% Generates data according to a model.
%
% Input:
%
% - model: model, as specified by 'generate_model_data'
% - N: sample size
%
% Output:
%
% - Z, X, Y: synthetic data

K = length(model.gZ{1});

dat.cov_Z = ones(K) * model.rho; for i = 1:K, dat.cov_Z(i, i) = 1; end
dat.Z = randn(N, K) * chol(dat.cov_Z);

dat.X = zeros(N, 1);
dat.Y = zeros(N, 1);
for n = 1:N
  z = dat.Z(n, :);
  x0 = z * model.gZ{1} + z * model.gZ{2} * z' + randn() * sqrt(model.v_x)-1;
  x = 1 / (1 + exp(-(x0)));
  %x= rand;
  y = [x sin(3*pi*x) sin(5*pi*x)] * model.fX + z * model.fZ{1} + z * model.fZ{2} * z' + randn() * sqrt(model.v_y);
  dat.X(n) = x;
  dat.Y(n) = y;
end

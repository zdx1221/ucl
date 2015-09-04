
function [model, dat] = generate_model_data(K, rho, v_x, v_y, N)

% generate_model_data:
%
% Generates a model for treatment X, outcome Y, given common causes Z.
%
% Common causes Z have dimensionality K, follow a Gaussian with zero mean
% and correlation matrix with off-diagonal entries 'rho'.
%
% Function X = s(g(Z) + epsilon_x) is a second order polynomial, and epsilon_x
% is Gaussian with zero mean and variance v_x. s(.) is the sigmoid function 
% that squashes a number to the [0, 1] interval: this makes evaluation
% easier, as we can plot Y as a function of X or do(X) within a bounded
% region.
%
% Function Y = f(X, Z) + epsilon_y is additive: f(X, Z) = f_1(X) + f_2(Z).
% f_1(X) is a cubic polynomial, f_2(Z) is a second order polynomial.
% epsilon_y is Gaussian with zero mean and variance v_y.
%
% From that, we have E[Y | do(X)] = f_1(X) + E[Z*fZ{2}*Z'] = f_1(x) + Tr(fZ{2}*cov_Z).
%
% Input:
%
% - K: number of elements in Z
% - rho: correlation for each pair of elements in Z
% - v_x: variance of error term epsilon_x
% - v_y: variance of error term epsilon_y
% - N: sample size
%
% Output:
%
% - gZ: cell structure of two entries, parameterizes g(Z) as follows
%   - gZ{1}: K by 1 vector, linear terms of g(Z)
%   - gZ{2}: K by K symmetric matrix, quadratic terms of g(Z)
%
%   That is, g(Z) = gZ{1} * Z + Z * gZ{2} * Z'
%
% - fX: 3 by 1 vector, contains linear, quadratic and cubic terms of f_1(X)
% - fZ{1}, fZ{2}: linear and quadratic terms of f_2(X)
%
%   That is, f(X, Z) = fX(1) * X + fX(2) * X^2 + fX(3) * X^3) + 
%                      fZ{1} * Z + Z * fZ{2} * Z'
%
% - dat: synthetic data

model.rho = rho;
model.v_x = v_x;
model.v_y = v_y;

model.gZ = cell(2, 1);
model.gZ{1} = randn(K, 1) ./ sqrt(K);
V = randn(K);
model.gZ{2} = V * V' ./ K^2;

model.fX = randn(3, 1) ./ [1; 5; 25];

model.fZ = cell(2, 1);
model.fZ{1} = randn(K, 1) ./ sqrt(K);
V = randn(K);
model.fZ{2} = V * V' ./ K^2;

dat = generate_data(model, N);

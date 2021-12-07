% Testing code for LRPCA
% Implemented by Jialin Liu @ Alibaba DAMO Academy
% Thanks the scaled GD codes: https://github.com/Titan-Tong/ScaledGD

%% Script Starts here
clear; 

%% Load Data
data_path = "./data/n1000_r5_alpha0.1.mat";
load(data_path);                            % load the RPCA problem
[n, r] = size(U_star);                      % size and rank
X_star = sparse(double(U_star * V_star'));  % ground truth
Y = sparse(double(Y_star));                 % observation

%% Load Model
model_path = "./trained_models/lrpcanet_alpha0.1.mat";
load(model_path);                           % load the trained parameters
zeta = ths * (1000/n) * (r/5);              % thresholds (adaptive to n,r)
eta  = step;                                % step sizes

%% Running LRPCA
[X,L,R] = LearnedRPCA(Y,r,X_star,zeta,eta); % x_star is only used for printing

%% Implementation of LRPCA
function [X,L,R] = LearnedRPCA(Y,r,X_star,zeta,eta)
    
    % preparation
    [~, T] = size(zeta);
    time_counter = 0;

    % Initialization
    tStart = tic;
    [U0, Sigma0, V0] = svds(Y - Thre(Y, zeta(1)), r);
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    time_counter = time_counter + toc(tStart);
    
    % main loop
    fprintf("===============LRPCA logs=============\n");
    for t = 1:(T-1)
        tStart = tic;
        X = L*R';
        S = Thre(Y - X, zeta(t+1)); 
        L_plus = L - eta(t+1)*(X+S-Y)*R/(R'*R + eps('double')*eye(r));
        R_plus = R - eta(t+1)*(X+S-Y)'*L/(L'*L + eps('double')*eye(r));
        L = L_plus;
        R = R_plus;
        time_counter = time_counter + toc(tStart);
        dist_X = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        fprintf("k: %d Err: %e Time: %f\n", t ,dist_X, time_counter);
    end
    fprintf("======================================\n");
end

%% Soft Thresholding
function S = Thre(S, theta)
S = sign(S) .* max(abs(S)-theta, 0.0);
end


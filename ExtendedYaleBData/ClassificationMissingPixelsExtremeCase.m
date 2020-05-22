%% 
% Script that compares the classification performance of sparse coding
% variants considered in "RobOMP: Robust variants of Orthogonal Matching 
% Pursuit for sparse representations" DOI: 10.7717/peerj-cs.192 (open access)
% Missing pixels noise scenario for feature dimension of 132 and fixed rate
% of missing pixels
% Author: Carlos Loza
% https://github.carlosloza/RobOMP
%
% Methodology:
% ------------
% 0. Important: Yale Face Database B subsets must be created beforehand,
% i.e. run "CreateSubsets.m" and follow instructions there
% 1. Subject-specific dictionaries are created based on selected subset(s)
% 2. Test set is created based on different selected subset(s)
% 3. Missing pixels noise is added to samples from test set: randomly
% selected pixels are replaced by samples from U[0, ymax] (uniform) where
% ymax is the largest intensity of the image in question
% 4. The following sparse coders are implemented:
%   - Orthogonal Matching Pursuit (OMP)
%   - Generalized OMP (with optional set of number of atoms per run)
%   - Correntropy Matching Pursuit
%   - Robust m-estimator-based variants of OMP: Fair, Cauchy, Huber, Tukey,
%   Welsh
% 5. Sample from test set is assigned to class (dictionary) that yields
% smaller L2-norm of residual
%
% Notes:
% ------
% - 38 subjects in total, 64 images per subject
% - Initial size of each image: 192 x 168
% - Several number of repetitions (runs) are allowed
% - This script replicates the results summarized in Table 5 of  RobOMP
% article
% - The results in the article took a random seed so the final outputs
% of this script might not exactly match the published results
% - Lastly, this new version implements a warm start of RobOMP by default, 
% i.e. the Huber solution is the initial solution for every RobOMP case
% Empirically, this initialization was not only proved to be more stable,
% but it also yielded better performance

close all
clearvars
clc

addpath('..')               % Assuming directories as in remote repo

K = 10;                     % Sparsity level
downsamp = 1/16;            % Downsample factor
misspixrate_v = 0.4;        % Set of missing pixels rates
TrainSubset = [1 2];        % Train subset(s) for class-dependent dictionaries
TestSubset = 3;             % Test subset(s)
N0_v = [2 3 5];             % Set of number of atoms extracted per iteration by gOMP
subj_v = [1:13 15:39];
nSub = length(subj_v);
nmisspix = length(misspixrate_v);

rng(34)                     % For reproducibility

nRep = 10;                  % Number of repetitions (runs)

%% Create Dictionary and extract clean test set
% KEY: Assuming directories as in remote repo, i.e. downloaded (from 
% http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/ExtYaleB.html) and unziped
% directory (Cropped Images version) should be placed in '../../Data'
% alongside created subsets. Otherwise, this will throw an error

fprintf('Creating dictionary... \n')
% Downsample
m = round(192*downsamp);
n = round(168*downsamp);
Dictionary(nSub) = struct();
for i = 1:nSub
    ct = 1;
    D = [];
    % Access training subset(s)
    for j = 1:length(TrainSubset)
        if subj_v(i) < 10
            DirFiles = ['../../Data/CroppedYale/yaleB0' num2str(subj_v(i)) '/Subset' num2str(TrainSubset(j))];
        else
            DirFiles = ['../../Data/CroppedYale/yaleB' num2str(subj_v(i)) '/Subset' num2str(TrainSubset(j))];
        end
        FileList = dir(fullfile(DirFiles, '*.pgm'));
        nImg = size(FileList,1);
        for k = 1:nImg
            % Read image
            auxImg = double(imread([DirFiles '/' FileList(k).name]));
            auxImg = auxImg(1:round(1/downsamp):end, 1:round(1/downsamp):end);   
            % Vectorize image
            auxVec = auxImg(:);
            % Unit-norm atoms
            auxVec = auxVec/(sqrt(sum(auxVec.^2)));
            D(:,ct) = auxVec;
            ct = ct + 1;
        end    
    end
    Dictionary(i).D = D;
end
fprintf('Done! \n')

fprintf('Creating test set... \n')
Test(nSub) = struct();
for i = 1:nSub
    ct = 1;
    Y = [];
    A = [];
    % Access test subset(s)
    for j = 1:length(TestSubset)
        if subj_v(i) < 10
            DirFiles = ['../../Data/CroppedYale/yaleB0' num2str(subj_v(i)) '/Subset' num2str(TestSubset(j))];
        else
            DirFiles = ['../../Data/CroppedYale/yaleB' num2str(subj_v(i)) '/Subset' num2str(TestSubset(j))];
        end
        FileList = dir(fullfile(DirFiles, '*.pgm'));
        nImg = size(FileList,1);
        for k = 1:nImg
            % Read image
            auxImg = double(imread([DirFiles '/' FileList(k).name]));
            auxImg = auxImg(1:round(1/downsamp):end, 1:round(1/downsamp):end);
            A(:,:,ct) = auxImg;
            % Vectorize image
            auxVec = auxImg(:);
            Y(:,ct) = auxVec;
            ct = ct + 1;
        end    
    end
    Test(i).Y = Y;
    Test(i).A = A;
end
fprintf('Done! \n')

%% Classification under missing pixels noise
ClassOMP = zeros(K, nRep, nmisspix);
ClassgOMP = zeros(K, nRep, nmisspix);
ClassCMP = zeros(K, nRep, nmisspix);
ClassCauchyOMP = zeros(K, nRep, nmisspix);
ClassFairOMP = zeros(K, nRep, nmisspix);
ClassHuberOMP = zeros(K, nRep, nmisspix);
ClassTukeyOMP = zeros(K, nRep, nmisspix);
ClassWelschOMP = zeros(K, nRep, nmisspix);
for misspix_i = 1:length(misspixrate_v)
    misspix = misspixrate_v(misspix_i);   
    for rep = 1:nRep
        clc
        fprintf('Missing pixels rate = %.2f, Repetition: %d \n', misspix, rep)
        % Add noise to Test set
        for sub_i = 1:nSub
            Y = Test(sub_i).Y;
            A = Test(sub_i).A;
            YNoise = zeros(size(Y));
            for i = 1:size(Y, 2)
                idxMiss = randperm(round(m*n), round(misspix*m*n));
                auxVecNoise = Y(:,i);
                % Adding noise here
                auxVecNoise(idxMiss) = max(auxVecNoise)*rand(round(misspix*m*n),1);
                % Vectorize noisy image
                YNoise(:, i) = auxVecNoise;
            end
            Test(sub_i).YNoise = YNoise;
        end
        
        % Classification        
        % OMP
        fprintf('OMP \n')
        ct_all = 0;
        ct_right = zeros(K, 1);
        for sub_i = 1:nSub
            YNoise = Test(sub_i).YNoise;
            nTest = size(YNoise,2);
            ct_all = ct_all + nTest;
            normM = zeros(nTest, nSub, K);
            for i = 1:nTest
                for sub_j = 1:nSub
                    D = Dictionary(sub_j).D;
                    [~, ~, ~, E] = gOMP(YNoise(:,i), D, 'N0', 1, 'nnonzero', K);
                    normM(i, sub_j, :) = sqrt(sum(E.^2, 1));
                end
            end
            [~, idxmin] = min(normM, [], 2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassOMP(:, rep, misspix_i) = ct_right./ct_all;
        
        % GOMP
        fprintf('gOMP \n')
        ct_all = 0;
        ct_right = zeros(K, 1);
        for sub_i = 1:nSub
            YNoise = Test(sub_i).YNoise;
            nTest = size(YNoise,2);
            ct_all = ct_all + nTest;
            normM = zeros(nTest, nSub, K);
            for i = 1:nTest
                for sub_j = 1:nSub
                    D = Dictionary(sub_j).D;
                    auxnorm = zeros(K, length(N0_v));
                    for j = 1:length(N0_v)
                        [~, ~, ~, ~, ~, E] = gOMP(YNoise(:,i), D, 'N0', N0_v(j), 'nnonzero', K);
                        auxnorm(:, j) = sqrt(sum(E.^2, 1));
                    end
                    normM(i, sub_j, :) = min(auxnorm, [], 2);
                end
            end
            [~, idxmin] = min(normM, [], 2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassgOMP(:, rep, misspix_i) = ct_right./ct_all;
        
        % CMP
        fprintf('CMP \n')
        ct_all = 0;
        ct_right = zeros(K, 1);
        for sub_i = 1:nSub
            YNoise = Test(sub_i).YNoise;
            nTest = size(YNoise,2);
            ct_all = ct_all + nTest;
            normM = zeros(nTest, nSub, K);
            for i = 1:nTest
                for sub_j = 1:nSub
                    D = Dictionary(sub_j).D;
                    [~, ~, ~, ~, E] = CMP(YNoise(:,i), D, 'nnonzero', K);                    
                    normM(i, sub_j, :) = sqrt(sum(E.^2, 1));
                end
            end
            [~, idxmin] = min(normM, [],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassCMP(:, rep, misspix_i) = ct_right./ct_all;
        
        % CauchyOMP
        fprintf('CauchyOMP \n')
        ct_all = 0;
        ct_right = zeros(K,1);
        for sub_i = 1:nSub
            YNoise = Test(sub_i).YNoise;
            nTest = size(YNoise,2);
            ct_all = ct_all + nTest;
            normM = zeros(nTest, nSub, K);
            for i = 1:nTest
                for sub_j = 1:nSub
                    D = Dictionary(sub_j).D;
                    [~, ~, ~, ~, E] = RobOMP(YNoise(:,i), D, 'm-est', 'Cauchy', 'nnonzero', K);
                    normM(i, sub_j, :) = sqrt(sum(E.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassCauchyOMP(:, rep, misspix_i) = ct_right./ct_all;
        
        % FairOMP
        fprintf('FairOMP \n')
        ct_all = 0;
        ct_right = zeros(K,1);
        for sub_i = 1:nSub
            YNoise = Test(sub_i).YNoise;
            nTest = size(YNoise,2);
            ct_all = ct_all + nTest;
            normM = zeros(nTest, nSub, K);
            for i = 1:nTest
                for sub_j = 1:nSub
                    D = Dictionary(sub_j).D;
                    [~, ~, ~, ~, E] = RobOMP(YNoise(:,i), D, 'm-est', 'Fair', 'nnonzero', K);
                    normM(i, sub_j, :) = sqrt(sum(E.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassFairOMP(:, rep, misspix_i) = ct_right./ct_all;
        
        % HuberOMP
        fprintf('HuberOMP \n')
        ct_all = 0;
        ct_right = zeros(K,1);
        for sub_i = 1:nSub
            YNoise = Test(sub_i).YNoise;
            nTest = size(YNoise,2);
            ct_all = ct_all + nTest;
            normM = zeros(nTest, nSub, K);
            for i = 1:nTest
                for sub_j = 1:nSub
                    D = Dictionary(sub_j).D;
                    [~, ~, ~, ~, E] = RobOMP(YNoise(:,i), D, 'm-est', 'Huber', 'nnonzero', K);
                    normM(i, sub_j, :) = sqrt(sum(E.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassHuberOMP(:, rep, misspix_i) = ct_right./ct_all;
        
        % TukeyOMP
        fprintf('TukeyOMP \n')
        ct_all = 0;
        ct_right = zeros(K,1);
        for sub_i = 1:nSub
            YNoise = Test(sub_i).YNoise;
            nTest = size(YNoise,2);
            ct_all = ct_all + nTest;
            normM = zeros(nTest, nSub, K);
            for i = 1:nTest
                for sub_j = 1:nSub
                    D = Dictionary(sub_j).D;
                    [~, ~, ~, ~, E] = RobOMP(YNoise(:,i), D, 'm-est', 'Tukey', 'nnonzero', K);
                    normM(i, sub_j, :) = sqrt(sum(E.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassTukeyOMP(:, rep, misspix_i) = ct_right./ct_all;
        
        % WelschOMP
        fprintf('WelschOMP \n')
        ct_all = 0;
        ct_right = zeros(K,1);
        for sub_i = 1:nSub
            YNoise = Test(sub_i).YNoise;
            nTest = size(YNoise,2);
            ct_all = ct_all + nTest;
            normM = zeros(nTest, nSub, K);
            for i = 1:nTest
                for sub_j = 1:nSub
                    D = Dictionary(sub_j).D;
                    [~, ~, ~, ~, E] = RobOMP(YNoise(:,i), D, 'm-est', 'Welsch', 'nnonzero', K);
                    normM(i, sub_j, :) = sqrt(sum(E.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassWelschOMP(:, rep, misspix_i) = ct_right./ct_all;
    end 
end
%% Plot results - Table 5
figure('units','normalized','outerposition',[0 0 1 1])
FontSize = 40;
FontSizeLegend = 30;
Linewidth = 5;
MarkerSize = 20;
K_v = 1:K;
plot(K_v, mean(ClassOMP, 2), '--+' , 'Color', [0 0 153]/255)
hold on
plot(K_v, mean(ClassgOMP, 2), '--x' , 'Color', [0 102 204]/255)
plot(K_v, mean(ClassCMP, 2), '--d' , 'Color', [0 0 0]/255)
plot(K_v, mean(ClassCauchyOMP, 2), '--^' , 'Color', [76 153 0]/255)
plot(K_v, mean(ClassFairOMP, 2), '--v' , 'Color', [102 0 102]/255)
plot(K_v, mean(ClassHuberOMP, 2), '-->' , 'Color', [255 0 0]/255)
plot(K_v, mean(ClassTukeyOMP, 2), '--<' , 'Color', [255 128 0]/255)
plot(K_v, mean(ClassWelschOMP, 2), '--o' , 'Color', [128 128 128]/255)
xlabel('Matching pursuit iteration (K)')
ylabel('Classification accuracy')
set(findall(gcf,'-property','FontSize'),'FontSize',FontSize)
set(findall(gcf,'-property','Linewidth'),'Linewidth',Linewidth)
set(findall(gcf,'-property','MarkerSize'),'MarkerSize',MarkerSize)
legend({'OMP','gOMP','CMP','Cauchy','Fair','Huber','Tukey','Welsch'},...
    'Location','Northwest','FontSize',FontSizeLegend);
ylim([0 1.1])
yticks(0:0.25:1)
xlim([-1.5 K])
xticks(K_v)
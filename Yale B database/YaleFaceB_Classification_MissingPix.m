%% Classification of Yale Face Database B 
% Noise as missing pixels
% All vectorized images of some subsets are used as a class-dependent 
% dictionary, i.e. one dictionary per subject
% Class is chosen according to minimal norm of residual
% Test set is another subset
% 38 subjects
% 64 pictures per subject
% Initial size of each image: 192 x 168

close all
clearvars
clc

K = 5;
downsamp = 0.5;
p_v = 0:0.1:1;                   % Missing pixels rate

SubsetTrain = [1 2];
SubsetTest = 3;

warning('off')

nRep = 10;

if K > 2
    N0_v = 2:ceil(K/2);             % GOMP parameter
else
    N0_v = K;
end

subj_v = [1:13 15:39];
nSub = length(subj_v);

%% Create Dictionary and extract test set

disp('Creating Dictionary')
m = round(192*downsamp);
n = round(168*downsamp);
Dictionary(nSub) = struct();
for i = 1:nSub
    ct = 1;
    D = [];
    for j = 1:length(SubsetTrain)
        if subj_v(i) < 10
            DirFiles = ['../Data/CroppedYale/yaleB0' num2str(subj_v(i)) '/Subset ' num2str(SubsetTrain(j))];
        else
            DirFiles = ['../Data/CroppedYale/yaleB' num2str(subj_v(i)) '/Subset ' num2str(SubsetTrain(j))];
        end
        FileList = dir(fullfile(DirFiles, '*.pgm'));
        nImg = size(FileList,1);
        for k = 1:nImg
            auxImg = double(imread([DirFiles '/' FileList(k).name]));
            auxImg = auxImg(1:round(1/downsamp):end, 1:round(1/downsamp):end);
            auxVec = auxImg(:);
            auxVec = auxVec/(sqrt(sum(auxVec.^2)));
            D(:,ct) = auxVec;
            ct = ct + 1;
        end    
    end
    Dictionary(i).D = D;
end
disp('Done!')

disp('Creating Test Set')
Test(nSub) = struct();
for i = 1:nSub
    ct = 1;
    Y = [];
    for j = 1:length(SubsetTest)
        if subj_v(i) < 10
            DirFiles = ['../Data/CroppedYale/yaleB0' num2str(subj_v(i)) '/Subset ' num2str(SubsetTest(j))];
        else
            DirFiles = ['../Data/CroppedYale/yaleB' num2str(subj_v(i)) '/Subset ' num2str(SubsetTest(j))];
        end
        FileList = dir(fullfile(DirFiles, '*.pgm'));
        nImg = size(FileList,1);
        for k = 1:nImg
            auxImg = double(imread([DirFiles '/' FileList(k).name]));
            auxImg = auxImg(1:round(1/downsamp):end, 1:round(1/downsamp):end);
            auxVec = auxImg(:);
            Y(:,ct) = auxVec;
            ct = ct + 1;
        end    
    end
    Test(i).Y = Y;
end
disp('Done!')

%%
for p_i = 1:length(p_v)
    p = p_v(p_i);
    ClassOMP = zeros(K, nRep);
    ClassGOMP = zeros(K, nRep);
    ClassCorrOMP = zeros(K, nRep);
    ClassCauchyOMP = zeros(K, nRep);
    ClassFairOMP = zeros(K, nRep);
    ClassHuberOMP = zeros(K, nRep);
    ClassTukeyOMP = zeros(K, nRep);
    ClassWelschOMP = zeros(K, nRep);
    for rep = 1:nRep
        clc
        disp(['p = ' num2str(p) ' Repetition ' num2str(rep)])
        % Create Dictionary per subject and add noise to Test set
        for sub_i = 1:nSub
            Y = Test(sub_i).Y;
            YNoise = zeros(size(Y));
            for i = 1:size(Y, 2)
                idxMiss = randperm(round(m*n), round(p*m*n));
                auxVecNoise = Y(:,i);
                auxVecNoise(idxMiss) = max(auxVecNoise)*rand(round(p*m*n),1);
                YNoise(:,i) = auxVecNoise;
            end
            Test(sub_i).YNoise = YNoise;
        end
        
        % Perform Classification
        
        % OMP
        tic
        disp('OMP')
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
                    [xOMP, eOMP] = gomp(YNoise(:,i), D, K, 1);
                    normM(i, sub_j, :) = sqrt(sum(eOMP.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassOMP(:,rep) = ct_right./ct_all;
        toc
        
        % GOMP
        tic
        disp('GOMP')
        ct_all = 0;
        ct_right = zeros(K,1);
        for sub_i = 1:nSub
            YNoise = Test(sub_i).YNoise;
            nTest = size(YNoise,2);
            ct_all = ct_all + nTest;
            normM = zeros(nTest, nSub, K);
            for i = 1:nTest
                for sub_j = 1:nSub
                    if sub_j == 11
                        df = 1;
                    end
                    D = Dictionary(sub_j).D;
                    auxnorm = zeros(K, length(N0_v));
                    for j = 1:length(N0_v)
                        [xGOMP, eGOMP] = gomp(YNoise(:,i), D, K, N0_v(j));
                        auxnorm(:, j) = sqrt(sum(eGOMP.^2,1));
                    end
                    normM(i, sub_j, :) = min(auxnorm, [], 2);
                end
                as = 1;
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassGOMP(:,rep) = ct_right./ct_all;
        toc
        
        % CorrOMP
        tic
        disp('CorrOMP')
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
                    [xOMP, eCorrOMP] = CorrOMP(YNoise(:,i), D, K);
                    normM(i, sub_j, :) = sqrt(sum(eCorrOMP.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassCorrOMP(:,rep) = ct_right./ct_all;
        toc
        
        % CauchyOMP
        tic
        disp('CauchyOMP')
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
                    [xOMP, eCauchyOMP] = CauchyOMP(YNoise(:,i), D, K);
                    normM(i, sub_j, :) = sqrt(sum(eCauchyOMP.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassCauchyOMP(:,rep) = ct_right./ct_all;
        toc
        
        % FairOMP
        tic
        disp('FairOMP')
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
                    [xOMP, eFairOMP] = FairOMP(YNoise(:,i), D, K);
                    normM(i, sub_j, :) = sqrt(sum(eFairOMP.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassFairOMP(:,rep) = ct_right./ct_all;
        toc
        
        % HuberOMP
        tic
        disp('HuberOMP')
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
                    [xOMP, eHuberOMP] = HuberOMP(YNoise(:,i), D, K);
                    normM(i, sub_j, :) = sqrt(sum(eHuberOMP.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassHuberOMP(:,rep) = ct_right./ct_all;
        toc
        
        % TukeyOMP
        tic
        disp('TukeyOMP')
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
                    [xOMP, eTukeyOMP] = TukeyOMP(YNoise(:,i), D, K);
                    normM(i, sub_j, :) = sqrt(sum(eTukeyOMP.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassTukeyOMP(:,rep) = ct_right./ct_all;
        toc
        
        % WelschOMP
        tic
        disp('WelschOMP')
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
                    [xOMP, eWelschOMP] = WelschOMP(YNoise(:,i), D, K);
                    normM(i, sub_j, :) = sqrt(sum(eWelschOMP.^2,1));
                end
            end
            [~, idxmin] = min(normM,[],2);
            ct_right = ct_right + squeeze(sum(idxmin == sub_i));
        end
        ClassWelschOMP(:,rep) = ct_right./ct_all;
        toc
    end
end

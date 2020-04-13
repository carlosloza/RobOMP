%%
% Script that creates subsets from Extended Yale B database according to
% azimuth and elevation angles of single light source
% 5 increasingly complex (and more challenging) subsets. This segmentation
% of the database has been used in other publications as well, e.g. 
% Wright et al., 2009
% Image processing toolbox needed. Alternatively, the python implementation
% can be used.
% Part of RobOMP package. "RobOMP: Robust variants of Orthogonal Matching 
% Pursuit for sparse representations" DOI: 10.7717/peerj-cs.192 (open access)
% Author: Carlos Loza
% https://github.carlosloza/RobOMP
% Note: Assuming directories as in remote repo, downloaded (from 
% http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/ExtYaleB.html) and unziped
% directory (Cropped Images version) should be placed in '../../Data'

close all
clearvars
clc

subj_v = [1:13 15:39];              % Subjects of database
nSub = length(subj_v);

fprintf('Creating subsets... \n')
for i = 1:nSub
    if subj_v(i) < 10
        DirFiles = ['../../Data/CroppedYale/yaleB0' num2str(subj_v(i))];
    else
        DirFiles = ['../../Data/CroppedYale/yaleB' num2str(subj_v(i))];
    end
    % Create directories for each subset
    for j = 1:5
        mkdir([DirFiles '/Subset' num2str(j)])
    end
    FileList = dir(fullfile(DirFiles, '*.pgm'));
    nImg = size(FileList,1);
    for j = 1:nImg
        auxImg = imread([DirFiles '/' FileList(j).name]);
        a1 = str2double(FileList(j).name(end-10:end-8));
        a2 = str2double(FileList(j).name(end-5:end-4));
        angl = round(sqrt(a1^2 + a2^2));
        if angl <= 12
            subs = '/Subset1/';
        elseif angl >= 13 && angl <= 25
            subs = '/Subset2/';
        elseif angl >= 26 && angl <= 54
            subs = '/Subset3/';
        elseif angl >= 55 && angl < 83
            subs = '/Subset4/';
        elseif angl >= 84
            subs = '/Subset5/';
        end
        imwrite(auxImg, [DirFiles subs FileList(j).name]);
    end
    sd = 1;
end
fprintf('Done! \n')
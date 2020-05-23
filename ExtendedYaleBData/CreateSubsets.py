"""
Script that creates subsets from Extended Yale B database according to
azimuth and elevation angles of single light source
5 increasingly complex (and more challenging) subsets. This segmentation
of the database has been used in other publications as well, e.g.
Wright et al., 2009
Part of RobOMP package. "RobOMP: Robust variants of Orthogonal Matching
Pursuit for sparse representations" DOI: 10.7717/peerj-cs.192 (open access)
Author: Carlos Loza
https://github.carlosloza/RobOMP
Note: Assuming directories as in remote repo, downloaded (from
http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/ExtYaleB.html) and unziped
directory (Cropped Images version) should be placed in '../../Data'
"""

import os
import glob

# Subjects of database
subj_v = list(range(1, 14)) + list(range(15, 40))
nSub = len(subj_v)

print('Creating subsets...')
for i in range(0, nSub):
    if subj_v[i] < 10:
        DirFiles = '../Data/CroppedYale/yaleB0' + str(subj_v[i])
    else:
        DirFiles = '../Data/CroppedYale/yaleB' + str(subj_v[i])
    # Create directories for each subset
    for j in range(1, 6):
        path = DirFiles + '/Subset' + str(j)
        os.mkdir(path)
    FileList = [f for f in glob.glob(DirFiles + '/*.pgm')]
    nImg = len(FileList)
    for j in range(0, nImg):
        if FileList[j][-11:-4] == 'Ambient':
            print('Ignore file')
        else:
            a1 = int(FileList[j][-11:-8])
            a2 = int(FileList[j][-6:-4])
            angl = round((a1**2 + a2**2)**0.5)
            if angl <= 12:
                subs = '/Subset1/'
            elif angl >= 13 and angl <= 25:
                subs = '/Subset2/'
            elif angl >= 26 and angl <= 54:
                subs = '/Subset3/'
            elif angl >= 55 and angl <= 83:
                subs = '/Subset4/'
            elif angl >= 84:
                subs = '/Subset5/'

            os.rename(FileList[j], DirFiles + subs + FileList[j][len(DirFiles)+1:])
print('Done!')
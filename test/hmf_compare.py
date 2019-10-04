
#!/bin/env python
import sys
import struct
import glob
import os.path
import re
import numpy as np
import pandas as pd
import eagle_IO.eagle_IO as E


h=0.6777

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):

    return [atoi(c) for c in re.split('(\d+)', text)]

def get_files(file_req):
    
    if np.isscalar(file_req):
        files = glob.glob(file_req)
        files.sort(key=natural_keys)
    else:
        files = []
        for i in file_req:
            tmp = glob.glob(i)
            tmp.sort(key=natural_keys)
            files.append(tmp)
        files = np.concatenate(files)
    return files

def read_binary_subfind(MyRank,start,finish,path):
    #Create empty lists
    GroupLen=[]
    GroupOffset=[]
    GroupMass=[]
    GroupPos=[]
    GroupM_Mean200=[]
    GroupR_Mean200=[]
    GroupM_Crit200=[]
    GroupR_Crit200=[]
    GroupM_TopHat200=[]
    GroupR_TopHat200=[]
    GroupContaminationLen=[]
    GroupContaminationMass=[]
    GroupNsubs=[]
    GroupFirstSub=[]

    SubgroupLen=[]
    SubgroupOffset=[]
    SubgroupParent=[]
    SubgroupMass=[]
    SubgroupPos=[]
    SubgroupVel=[]
    SubgroupCM=[]
    SubgroupSpin=[]
    SubgroupVelDisp=[]
    SubgroupVmax=[]
    SubgroupVmaxRad=[]
    SubgroupHalfMass=[]
    SubgroupMostBoundID=[]
    SubgroupGrNr=[]

    # Read over defined range
    for i in range(start,finish+1):
        fpath=path+str(i)
        f=open(fpath,'rb')
        fileContent=struct.unpack('<IIIQIII',f.read(32))
        NGroups = fileContent[0]
        TotNGroups = fileContent[1]
        Nids = fileContent[2]
        TotNids = fileContent[3]
        NTask = fileContent[4]
        Nsubgroups = fileContent[5]
        TotNSubgroups = fileContent[6]

        if MyRank == 0 and i == start:
            print (" FOF groups: "+str(TotNGroups)+"  Subgroups: "+str(TotNSubgroups))

        # extend arrays for each file
        if NGroups>0:
            GroupLen.extend(struct.unpack('i'*NGroups,f.read(NGroups*4)))
            GroupOffset.extend(struct.unpack('i'*NGroups,f.read(NGroups*4)))
            GroupMass.extend(struct.unpack('f'*NGroups,f.read(NGroups*4)))
            GroupPos.extend(struct.unpack('f'*3*NGroups,f.read(3*NGroups*4)))
            GroupM_Mean200.extend(struct.unpack('f'*NGroups,f.read(NGroups*4)))
            GroupR_Mean200.extend(struct.unpack('f'*NGroups,f.read(NGroups*4)))
            GroupM_Crit200.extend(struct.unpack('f'*NGroups,f.read(NGroups*4)))
            GroupR_Crit200.extend(struct.unpack('f'*NGroups,f.read(NGroups*4)))
            GroupM_TopHat200.extend(struct.unpack('f'*NGroups,f.read(NGroups*4)))
            GroupR_TopHat200.extend(struct.unpack('f'*NGroups,f.read(NGroups*4)))
            GroupContaminationLen.extend(struct.unpack('i'*NGroups,f.read(NGroups*4)))
            GroupContaminationMass.extend(struct.unpack('f'*NGroups,f.read(NGroups*4)))
            GroupNsubs.extend(struct.unpack('i'*NGroups,f.read(NGroups*4)))
            GroupFirstSub.extend(struct.unpack('i'*NGroups,f.read(NGroups*4)))
        
        if Nsubgroups>0:
            SubgroupLen.extend(struct.unpack('i'*Nsubgroups,f.read(Nsubgroups*4)))
            SubgroupOffset.extend(struct.unpack('i'*Nsubgroups,f.read(Nsubgroups*4)))
            SubgroupParent.extend(struct.unpack('i'*Nsubgroups,f.read(Nsubgroups*4)))
            SubgroupMass.extend(struct.unpack('f'*Nsubgroups,f.read(Nsubgroups*4)))
            SubgroupPos.extend(struct.unpack('f'*3*Nsubgroups,f.read(3*Nsubgroups*4)))
            SubgroupVel.extend(struct.unpack('f'*3*Nsubgroups,f.read(3*Nsubgroups*4)))
            SubgroupCM.extend(struct.unpack('f'*3*Nsubgroups,f.read(3*Nsubgroups*4)))
            SubgroupSpin.extend(struct.unpack('f'*3*Nsubgroups,f.read(3*Nsubgroups*4)))
            SubgroupVelDisp.extend(struct.unpack('f'*Nsubgroups,f.read(Nsubgroups*4)))
            SubgroupVmax.extend(struct.unpack('f'*Nsubgroups,f.read(Nsubgroups*4)))
            SubgroupVmaxRad.extend(struct.unpack('f'*Nsubgroups,f.read(Nsubgroups*4)))
            SubgroupHalfMass.extend(struct.unpack('f'*Nsubgroups,f.read(Nsubgroups*4)))
            SubgroupMostBoundID.extend(struct.unpack('l'*Nsubgroups,f.read(Nsubgroups*8)))
            SubgroupGrNr.extend(struct.unpack('i'*Nsubgroups,f.read(Nsubgroups*4)))
        
        f.close()

    return np.array(GroupM_Crit200)*1e10/h, np.array(SubgroupMass)*1e10/h
    
file_req = '/cosma5/data/dp004/PROJECTS/Eagle/dm_eagle_volume_L3200/groups_002/subhalo_tab_002*'
path = '/cosma5/data/dp004/PROJECTS/Eagle/dm_eagle_volume_L3200/groups_002/subhalo_tab_002.'
files = get_files(file_req)
P_Mcrit200 = np.empty(714660)
P_Msub = np.empty(707868)
jj = 0
kk = 0
for ii in range(len(files)):

    tmp1, tmp2 = read_binary_subfind(0,ii,ii,path)
    tlen1 = len(tmp1)
    tlen2 = len(tmp2)
    P_Mcrit200[jj:jj+tlen1] = tmp1
    P_Msub[kk:kk+tlen2] = tmp2
    jj+=tlen1
    kk+=tlen2


bins = np.arange(8, 17, 0.3)
bincen = (bins[1:]+bins[:-1])/2.
binwidth=bins[1:]-bins[:-1]

phist1, edges = np.histogram(np.log10(P_Mcrit200[P_Mcrit200>0]), bins)
phist1 = phist1/(binwidth*(3200**3))

phist2, edges = np.histogram(np.log10(P_Msub[P_Msub>0]), bins)
phist2 = phist2/(binwidth*(3200**3))


al = [15,31,33] 
sims = np.arange(0,38)
sims = np.delete(sims,al)
tag='011_z004p770'

df = pd.read_csv('../grid_weights.txt')
weight_grids = np.array(df['weights'])

df = pd.read_csv('../weights_cdf.txt')
weight_cdf = np.array(df['weights'])

out_grids = np.zeros(len(bincen))
out_cdf = np.zeros(len(bincen))

for ii, jj in enumerate(sims):
    
    num = str(jj)
    if len(num) == 1:
        num =  '0'+num
    sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_{}/data'.format(num) 
    Mcrit200 = E.read_array('SUBFIND', sim, tag, '/FOF/Group_M_Crit200', numThreads=4, noH=True, physicalUnits=True)*1e10
    
    hist, edges = np.histogram(np.log10(Mcrit200), bins)
    
    out_grids+=hist*weight_grids[ii]/(binwidth*((4/3)*np.pi*(14/h)**3))
    out_cdf+=hist*weight_cdf[ii]/(binwidth*((4/3)*np.pi*(14/h)**3))
    

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 10), sharex=True, sharey=True, facecolor='w', edgecolor='k') 

axs.plot(bincen, np.log10(phist1), color = 'black', ls = 'solid', lw = 3, alpha = 0.5, label = r'3.2 Gpc box _ Mcrit200') 
axs.plot(bincen, np.log10(phist2), color = 'brown', ls = 'solid', lw = 3, alpha = 0.5, label = r'3.2 Gpc box - Msub')  
axs.plot(bincen, np.log10(out_grids), color = 'red', ls = 'solid', lw = 3, alpha = 0.5, label = r'Grid weights')  
axs.plot(bincen, np.log10(out_cdf), color = 'blue', ls = 'solid', lw = 3, alpha = 0.5, label = r'cdf weights')  


axs.set_xlabel(r'M$_{crit200}$/M$_{\odot}$', fontsize = 16)
axs.set_ylabel(r'log$_{10}$($\phi$/(cMpc$^{-3}$M$_{\odot}^{-1}$dex$^{-1}$))', fontsize = 16) 
axs.grid(which='major', alpha = 0.7, linestyle = '-')
axs.grid(which='minor', alpha = 0.4, linestyle = '--')
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(18)
axs.legend(fontsize = 16, loc = 3, frameon = False)
plt.savefig('Mcrit_hist_comp.png')
plt.show()


"""

    Script to load the delta values created using the chosen gridsize and selects
    regions for resimulation

"""

import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.spatial.distance import cdist
from methods import simulation, grid_coordinates

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

req = float(sys.argv[1])   #receiving the required sigma region at runtime

sim = simulation()
sim.show()

dat = np.load('conv_output.npz')['delta']

delta_log = np.log(dat)
mu, sig = norm.fit(delta_log.flatten())

import pickle as pcl
pcl.dump(norm,open('log_delta_fit.p','wb'))

def selection(x,centre,sig,dx,tol=5e-3):

    mask = (x-centre) < (dx+tol)*sig
    mask = mask & ((x-centre) > (dx-tol)*sig)
    return np.array(np.where(mask))


# to select regions of a given overdensity, set `dx`
# choose `tol` to adjust how near you select

## get overlapping regions given the coordinates to compare against
def rem_overlapping_regions(df, coords, sim, same):
    """

    Compute the distances across points given the dataframe or the coordinates array.
    True option best used when removing overlapping regions within the same dataframe
    right after computing their x, y and z coordinates.    
    
    Args:
        df: dataframe object containing x, y and z coordinates
        
        coords(array or dataframe): contains the coordinates to be compared the distance to. Given as 
        2d array when comparing to the same sigma selection as the x, y an z already available.
        If given as dataframe the coordinates need to taken from there.
        
        sim (object): simulation class
        
        same (boolean): True -> coords is an array and contains coordinates of the same dataframe 
        
    
    """
    
    tmp = df
    arr = np.ones(len(tmp))*np.nan
    
    if same:
        print ("Computing distances across same coordinates")
        dist = cdist(coords,coords,'euclidean')
        for i in range(len(dist)):
        
            ok = np.where(dist[i] < 2*sim.r)[0]
            ok = ok[ok > i]
            if np.sum(ok) > 0.:
                arr[ok] = ok
    
    else:
        print ("Computing distances across coordinates from dataframes")
        sel_coords = np.zeros((len(tmp), 3))
        comp_coords = np.zeros((len(coords), 3))
        sel_coords[:,0], sel_coords[:,1], sel_coords[:,2] = tmp['x'], tmp['y'], tmp['z']
        comp_coords[:,0], comp_coords[:,1], comp_coords[:,2] = coords['x'], coords['y'], coords['z']
        dist = cdist(comp_coords,sel_coords,'euclidean')
        for i in range(len(dist)):
        
            ok = np.where(dist[i] < 2*sim.r)[0]
            if np.sum(ok) > 0.:
                arr[ok] = ok
    
    
    arr = arr[~np.isnan(arr)]
    tmp = tmp.drop(arr).reset_index(drop=True)    

    return tmp

def get_highest_overdensity(delta_log, sim, mu, sig):

    #For the densest region it is much better to sort them from highest to lowest overdensity
    sel = np.where(delta_log > 4.95*sig + mu)
    sel_hdel = np.zeros((len(sel[0]), 3))
    sel_hdel[:,0], sel_hdel[:,1], sel_hdel[:,2] = sel[0], sel[1], sel[2]
    coords_hdel = sel_hdel * sim.conv

    ## Dataframe contents of the highest overdensity regions:
    df_hdel = pd.DataFrame({'x': coords_hdel[:,0], 'y': coords_hdel[:,1], 'z': coords_hdel[:,2]})
    df_hdel['log(1+delta)'] = delta_log[sel] 
    df_hdel['delta'] = np.exp(df_hdel['log(1+delta)']) - 1
    df_hdel['sigma'] = (delta_log[sel] - mu)/sig 
    df_hdel = (df_hdel.sort_values(by=['sigma'], ascending=False)).reset_index(drop=True)
    coords_hdel[:,0], coords_hdel[:,1], coords_hdel[:,2] = df_hdel['x'], df_hdel['y'], df_hdel['z']

    #Removing overlapping regions overlapping within the overdense ones
    df_hdel = rem_overlapping_regions(df_hdel, coords_hdel, sim, True)   
    print ("Highest overdensity regions > 4.95 sigma: ")
    print (df_hdel.head())
    
    return df_hdel


#select required-sigma regions -> req*sigma +/- tol:
sel_sigma = selection(delta_log, centre=mu, sig=sig, dx=req, tol=1e-3).T

#Randomly select 100 such regions from here. This selection would remain 
#the same as long as the set 'tol' remains the unchanged with same seed '680010'.  
rand = np.random.RandomState(seed=680010).permutation(len(sel_sigma)-1)[:100]
sel_sigma = sel_sigma[rand]
coords_sigma = sel_sigma * sim.conv
sel = [(sel_sigma[:,0]), (sel_sigma[:,1]), (sel_sigma[:,2])]

## Dataframe contents of the required region:
df_sigma = pd.DataFrame({'x': coords_sigma[:,0], 'y': coords_sigma[:,1], 'z': coords_sigma[:,2]})
df_sigma['log(1+delta)'] = delta_log[sel] 
df_sigma['delta'] = np.exp(df_sigma['log(1+delta)']) - 1
df_sigma['sigma'] = (delta_log[sel] - mu)/sig 

#Removing overlapping regions within this overdensity range
df_nonoverlap =  rem_overlapping_regions(df_sigma, coords_sigma, sim, True)   

####Removing regions overlapping with the overdense region, now reduntant due to alrady calculating####
#df_new = rem_overlapping_regions(df_nonoverlap, df_hdel, sim, False)

#Getting the required new regions
df_old = pd.read_table('GEAGLE_regions.txt', delim_whitespace=True)
df_new = rem_overlapping_regions(df_nonoverlap, df_old, sim, False)
print ("Required overdensity regime {} sigma +/- 1e-4: ".format(req))
print (df_new.head())


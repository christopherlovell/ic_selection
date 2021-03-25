import numpy as np

class simulation():

    def __init__(self):
        self.h = 0.6777
        self.boxsize = 3200 * self.h
        self.r = 14. 
        self.grid = 1200
        self.conv = (self.boxsize/self.grid)

    def show(self):
        print('Hubble parameter:',self.h)
        print('Boxsize (cMpc/h):',self.boxsize)
        print('Grid dimensions:',self.grid,'^3')



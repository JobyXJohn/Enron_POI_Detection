# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:15:47 2016

@author: joby
"""

# script to plot and visualize data: 
def plot_features(data,data_dict,features_list,rel_feat2,rel_feat3):
    sdata=data
    poi_sdata = sdata[sdata[:,0]==1.0,]
    non_sdata = sdata[sdata[:,0]==0.0,]
    
    import numpy
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.plot(poi_sdata[:,rel_feat2[0]],poi_sdata[:,rel_feat2[1]],'rs',
                non_sdata[:,rel_feat2[0]],non_sdata[:,rel_feat2[1]],'bs')
    plt.xlabel(features_list[rel_feat2[0]])
    plt.ylabel(features_list[rel_feat2[1]])
    plt.show()  
    
    # 3D plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = ['b','r']    
    for ii in [0,1]:
        xs = sdata[sdata[:,0]==ii,rel_feat3[0]]
        ys = sdata[sdata[:,0]==ii,rel_feat3[1]]
        zs = sdata[sdata[:,0]==ii,rel_feat3[2]]
        ax.scatter(xs, ys, zs, c=c[ii], marker='s')
    ax.set_xlabel(features_list[rel_feat3[0]])
    ax.set_ylabel(features_list[rel_feat3[1]])
    ax.set_zlabel(features_list[rel_feat3[2]])
    plt.show()
    

#    
######## Following Line used to identify outliers.
    #### Need to load variables to workspace and only run the following lines.
#ii=0    
#for item in  data_dict:
#    if data_dict[item]['total_payments']>1e8 and data_dict[item]['total_payments']!='NaN':
#        print item, data_dict[item]['bonus']
#        ii=ii+1 # count the number of outliers
#print ii        
#
#from itertools import islice
#def take(n, iterable):
#    "Return first n items of the iterable as a list"
#    return list(islice(iterable, n))
#n_items = take(20, data_dict.iteritems())    
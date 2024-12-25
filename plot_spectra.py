"""
This code (Nelson + Kieran) does lots and lots of strange things. 

TO RUN: 
    MAC:     python3 eval_csv/plot_spectra.py -a ~kieranrudd/Documents/MCDDM/eval_csv/Adjusters -p ~kieranrudd/Documents/MCDDM/eval_csv/DATA
    WINDOWS:    ...

    See lines ~30 for explaination of directories
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import pandas as pd
from scipy.signal import find_peaks
import glob
from glob import glob
import argparse
import time
from PIL import Image

sns.set_theme()

scale = 1                           #IMPORTANT. - alter this according to the degree increments taken during expereiments.
sensor_range = 2.65                      #observable range of sensor.

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, help='Directory of .csv files which are being evaluated')
parser.add_argument('-a', '--adjusters', type=str, help='Directory of adjusters')
args = parser.parse_args()
datapath = args.path
adjusterspath = args.adjusters
filename = glob(datapath + '/*.csv')[0]
csv_path = glob(datapath + '/*.csv')
csv_path.sort()

"""""
#to see files being scanned. (helped me check the order they were being scanned in!)
csv_files = glob(datapath + '/*.csv')
csv_files.sort()
for csv_file in csv_files:
  print('File read:',csv_file) 
"""

raw_path = pd.read_csv(adjusterspath + '/raw_spectra.csv', names=['Column1', 'Column2'])
x_raw_spectra = raw_path['Column1'][:]
y_raw_spectra = raw_path['Column2'][:]
raw_spectra_max = max(y_raw_spectra)
raw_spectra_adj = [val / raw_spectra_max for val in y_raw_spectra]
raw_spectra_adj[:] = [1/val for val in raw_spectra_adj]

dark_path = pd.read_csv(adjusterspath + '/dark_spectra.csv', names=['Column1', 'Column2'])
x_dark_spectra = dark_path['Column1'][:]
y_dark_spectra = dark_path['Column2'][:]
dark_spectra_max = max(y_dark_spectra)
dark_spectra_adj = [val / dark_spectra_max for val in y_dark_spectra]
dark_spectra_adj[:] = [1/val for val in dark_spectra_adj]

af = pd.read_csv(adjusterspath + '/spectra_accel.csv', names=['Column1','Column2'])
range_points = af['Column1'][:]
physical_points = af['Column2'][:]


plt.plot(physical_points, dark_spectra_adj, label='Dark Inverted')
plt.plot(physical_points, raw_spectra_adj, label='Raw Inverted')
plt.plot(physical_points, y_dark_spectra, label='Dark')
plt.plot(physical_points, y_raw_spectra, label='Raw')
plt.legend(loc="upper left")
plt.xlabel('Wavelength Range (mm)')
plt.ylabel('Intensity (%)')
plt.xlim(0,2.65)
plt.ylim(0,100)
plt.savefig(datapath + '/Evaluations/spectra_adj.svg', dpi=600)         #spectra adjustment plot
plt.show()


#"""
#####################################################################
#"""

df1 = pd.read_csv(filename, names=['Column1', 'Column2']) 
x_range = np.linspace(0, 100, 512) #512 data points per measurement
y_raw = df1['Column1'][:]
y_raw /= 203.8
y_dark = df1['Column2'][:]
y_dark /= 203.8
y_dark[:] = [x * y for x, y in zip(y_dark, dark_spectra_adj)]
fig, ax = plt.subplots(figsize=(4.8, 5.0)) 
plt.plot(x_range, y_raw, color='tab:orange', label='raw') 
plt.plot(x_range, y_dark, color='tab:blue', label='dark')
plt.legend() 
plt.xlabel('range (%)')
plt.ylabel('intensity (%)')
plt.savefig(datapath + '/Evaluations/raw_dark.svg', dpi=600) 
plt.show() 

#"""
#####################################################################
#"""

cm = plt.cm.Blues(np.linspace(0.3, 1, len(csv_path))) 
fig, ax = plt.subplots(figsize=(4.8, 5.0))
ax.set_prop_cycle('color', list(cm)) 
for f in csv_path: 
    df = pd.read_csv(f, names=['Column1', 'Column2'])
    y_raw = df['Column1'][:]
    y_raw /= 40.95
    y_raw[:] = [x * y for x, y in zip(y_raw, dark_spectra_adj)]
    plt.plot(physical_points, y_raw)
plt.xlabel('range (%)') 
plt.ylabel('intensity (%)')
plt.savefig(datapath + '/Evaluations/raw_spectra_overlaid.svg', dpi=600)
plt.show()

#"""
#####################################################################
#"""

all_peaks = [] 
filenames = [] 
peak_nums_at_given_angle = []

cm = plt.cm.BuPu(np.linspace(0.3, 1, len(csv_path)))    #colourmap
fig, ax = plt.subplots(figsize=(7, 20))
ax.set_prop_cycle('color', list(cm))

for i, f in enumerate(csv_path):
    print("Processing file:", f)
    df = pd.read_csv(f, names=['Raw', 'Dark'])
    y_dark = df['Dark'][:]
    y_dark /= 203.8
    y_dark = np.clip(y_dark, 0, 100) #clips data to within 0 - 100; any value outside of this becomes 0 or 100 respectively.
    y_dark[:] = [a * b for a, b in zip(y_dark, dark_spectra_adj)] #normalise

    peaks, _ = find_peaks(y_dark, [0.7, None], prominence=[0.6, None], width=[0.5, None]) #returns array of peaks (ndarray) and properties (as a dict)
    if i == 0:
        all_peaks.append(peaks)  # Append for the first iteration only, otherwise fails at 'peaks = all_peaks[i-1]'
    else:
        if len(peaks) < 2:
            peaks = all_peaks[i-1]
        all_peaks.append(peaks) #appends identified peak indices to the all_peaks list 
    filenames.append(f) #adds current file path, f, to the filenames list. 
    #data_peaks[i] = x_range[r_peaks[:2]] #populates data_peaks array with the first two elements (0, 1  :2) from the x_range array corresponding to the identified r_peaks
    peak_nums_at_given_angle.append(len(peaks))
    ax.plot(physical_points, y_dark)
    ax.figure.set_size_inches(5.5,5)
    
    
ax.set_xlabel('Range (mm)', fontsize='large', fontweight='bold')
ax.set_ylabel('Intensity (%)', fontsize='large', fontweight='bold')
#ax.set_ylim(0, 40)
ax.set_xlim(0,2.5)
#ax.set_yticks(np.arange(0,12,2))
ax.set_xticks(np.arange(0,2.5,0.5))
plt.savefig(datapath + '/Evaluations/dark_spectra_overlaid.svg', dpi=600)
plt.show()

#"""
#####################################################################
#"""

#xpos
dividor = 512 / sensor_range
peak_xpos_1D = []
for sub_array in all_peaks:
    for number in sub_array:
        peak_xpos_1D.append(number)

peak_xpos_1D[:] = [x/dividor for x in peak_xpos_1D]
peak_xpos_1D[:] = [0.1055*x*x*x -0.7024*x*x + 2.1158*x for x in peak_xpos_1D] #accounts for 'spectra acceleration'

min_x = min(peak_xpos_1D)
min_x_rnd = str(round(min_x, 5))
print('\nSpectra empty space:  ',min_x_rnd, 'mm')
peak_xpos_1D[:] = [x - min_x for x in peak_xpos_1D]

#rads
i=1
peak_degs_1D = []
while i <= len(peak_nums_at_given_angle):
    num = peak_nums_at_given_angle[i-1]
    num_temp = num
    while num_temp > 0:
        peak_degs_1D.append(i)
        num_temp = num_temp -1
    i = i + 1



peak_degs_1D[:] = [x * scale for x in peak_degs_1D] #no of measurements into degrees. 

peak_rads_1D = [x * np.pi/180 for x in peak_degs_1D]

plt.figure(figsize=(5.5,5))
plt.scatter(peak_xpos_1D,peak_degs_1D, s=1.4)
plt.xlabel('Range (mm)', fontsize='large', fontweight='bold')
plt.ylabel('Angle (°)', fontsize='large', fontweight='bold')
plt.xlim(0,1.6)
plt.ylim(0,365)
plt.savefig(datapath + '/Evaluations/waterfall.svg')
plt.show()

#"""
#####################################################################
#"""

sorted_peak_xpos_1D = sorted(peak_xpos_1D)
avg = (sorted_peak_xpos_1D[-10])/2
avg_rnd = str(round(avg, 5))
print ('Radius of microtubing:', avg_rnd, 'mm\n')

peak_rads_1D_1234 = []
for i, value in enumerate(peak_xpos_1D):
  if value < avg:
    peak_rads_1D_1234.append(peak_rads_1D[i] + np.pi)
  else:
    peak_rads_1D_1234.append(peak_rads_1D[i])
peak_xpos_1D_1234 = peak_xpos_1D[:]

peak_xpos_1D_1234[:] = [x - avg for x in peak_xpos_1D_1234]
peak_xpos_1D_1234[:] = [-i if i <0 else i for i in peak_xpos_1D_1234]
peak_rads_1D_1234[:] = [i + np.pi for i in peak_rads_1D_1234]

ax = plt.subplot(111, projection='polar')
ax.plot(peak_rads_1D_1234, peak_xpos_1D_1234, linestyle = 'None', color='black', marker='.', markersize='2', markerfacecolor='black')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1) # anticlockwise
ax.grid(True)
ax.tick_params(axis='y', colors='crimson')
ax.figure.set_size_inches(6,6)
plt.yticks(np.arange(0,1.1,0.1))
plt.ylim(0,0.8)
plt.savefig(datapath + '/Evaluations/visual.svg')
plt.show()

#"""
#####################################################################
#"""

peak_xpos_1D_34 = [x - avg for x in peak_xpos_1D if x >= 0.8]
peak_rads_1D_34 = [peak_rads_1D[i] for i, x in enumerate(peak_xpos_1D) if x >= 0.8]
peak_rads_1D_34 = [i + np.pi for i in peak_rads_1D_34]

ax = plt.subplot(111, projection='polar')
ax.plot(peak_rads_1D_34, peak_xpos_1D_34, linestyle = 'None', color='black', marker='.', markersize='2', markerfacecolor='black')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1) # clockwise
ax.grid(True)
ax.tick_params(axis='y', colors='crimson')
plt.yticks(np.arange(0,1.1,0.1))
plt.ylim(0,0.8)
ax.figure.set_size_inches(6,6)
plt.savefig(datapath + '/Evaluations/visual_p34.svg')
plt.show()

#"""
#####################################################################
#"""

peak_xpos_1D_12 = [x - avg for x in peak_xpos_1D if x <= avg]
peak_xpos_1D_12 = [-i for i in peak_xpos_1D_12]
peak_rads_1D_12 = [peak_rads_1D[i] for i, x in enumerate(peak_xpos_1D) if x <= avg]
#peak_rads_1D_12[:] = [i + np.pi for i in peak_rads_1D_12]

ax = plt.subplot(111, projection='polar')
ax.plot(peak_rads_1D_12, peak_xpos_1D_12, linestyle = 'None', color='black', marker='.', markersize='2', markerfacecolor='black')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1) # clockwise
ax.grid(True)
ax.tick_params(axis='y', colors='crimson')
plt.yticks(np.arange(0,1.1,0.1))
plt.ylim(0,0.8)
ax.figure.set_size_inches(6,6)
plt.savefig(datapath + '/Evaluations/visual_p12.svg')
plt.show()


#"""
#####################################################################
#"""

plt.scatter(peak_xpos_1D_12,peak_rads_1D_12, s=1.4)
plt.xlabel('range (mm)')
plt.ylabel('angle')
plt.xlim(0,1)
plt.ylim(3,10)
plt.yticks(np.arange(0,10,1))
plt.savefig(datapath + '/Evaluations/waterfall_p12.svg')
plt.show()

#"""
#####################################################################
#"""

mean_12 = np.mean(peak_xpos_1D_12)
#print('mean_12 #1:',mean_12)
peak_xpos_1D_p2 = [i for i in peak_xpos_1D_12 if i<mean_12]
peak_xpos_1D_p1 = [i for i in peak_xpos_1D_12 if i>mean_12]

q1_p2 = np.percentile(peak_xpos_1D_p2, 25)
q3_p2 = np.percentile(peak_xpos_1D_p2, 75)
iqr_p2 = q3_p2 - q1_p2
lwr_bou_p2 = q1_p2 - 1.5 * iqr_p2
upp_bou_p2 = q3_p2 + 1.5 * iqr_p2

q1_p1 = np.percentile(peak_xpos_1D_p1, 25)
q3_p1 = np.percentile(peak_xpos_1D_p1, 75)
iqr_p1 = q3_p1 - q1_p1
lwr_bou_p1 = q1_p1 - 1.5 * iqr_p1
upp_bou_p1 = q3_p1 + 1.5 * iqr_p1

for i, val in enumerate(peak_xpos_1D_12):
    if val < lwr_bou_p2:
        peak_xpos_1D_12.pop(i)
        peak_rads_1D_12.pop(i)
    if val > upp_bou_p1:
        peak_xpos_1D_12.pop(i)
        peak_rads_1D_12.pop(i)
    if upp_bou_p2 < val < lwr_bou_p1:
        peak_xpos_1D_12.pop(i)
        peak_rads_1D_12.pop(i)

plt.scatter(peak_xpos_1D_12,peak_rads_1D_12, s=1.4)
plt.xlabel('range (mm)')
plt.ylabel('angle')
plt.xlim(0,1)
plt.ylim(3,10)
plt.yticks(np.arange(0,10,1))
plt.savefig(datapath + '/Evaluations/waterfall_p12_no_outliers.svg')
plt.show()

#"""
#####################################################################
#"""

mean_12 = np.mean(peak_xpos_1D_12)
#print('mean_12 #2:',mean_12, '\n')
peak_xpos_1D_p2 = [x for x in peak_xpos_1D_12 if x<mean_12]
peak_xpos_1D_p1 = [x for x in peak_xpos_1D_12 if x>mean_12]
peak_rads_1D_p2 = [peak_rads_1D_12[i] for i, x in enumerate(peak_xpos_1D_12) if x<mean_12]
peak_rads_1D_p1 = [peak_rads_1D_12[i] for i, x in enumerate(peak_xpos_1D_12) if x>mean_12]


window_size = int(11)   #must be odd number; 3 or over.
window_shift = (window_size-1)/2
peak_xpos_1D_p2_extended = peak_xpos_1D_p2.copy()
peak_xpos_1D_p1_extended = peak_xpos_1D_p1.copy()
for i in range(int(window_shift)):
    peak_xpos_1D_p2_extended.insert(0, peak_xpos_1D_p2[-i-1])
    peak_xpos_1D_p1_extended.insert(0, peak_xpos_1D_p1[-i-1])
    peak_xpos_1D_p2_extended.append(peak_xpos_1D_p2[i])
    peak_xpos_1D_p1_extended.append(peak_xpos_1D_p1[i])

i_1 = 0
peak_xpos_1D_p2_moving_average = []
while i_1 < len(peak_xpos_1D_p2_extended) - window_size + 1:
    window = peak_xpos_1D_p2_extended[i_1:i_1+window_size]
    window_average = round(sum(window) / window_size, 2)
    peak_xpos_1D_p2_moving_average.append(window_average)
    i_1 += 1

i_1 = 0
peak_xpos_1D_p1_moving_average = []
while i_1 < len(peak_xpos_1D_p1_extended) - window_size + 1:
    window = peak_xpos_1D_p1_extended[i_1:i_1+window_size]
    window_average = round(sum(window) / window_size, 2)
    peak_xpos_1D_p1_moving_average.append(window_average)
    i_1 += 1

peak_xpos_1D_12_moving_average = np.concatenate((peak_xpos_1D_p1_moving_average,peak_xpos_1D_p2_moving_average))
peak_rads_1D_12_moving_average = np.concatenate((peak_rads_1D_p1,peak_rads_1D_p2))

plt.scatter(peak_xpos_1D_12_moving_average, peak_rads_1D_12_moving_average, s=1.4)
plt.xlabel('range (mm)')
plt.ylabel('angle')
plt.xlim(0,1)
plt.ylim(0,10)
plt.yticks(np.arange(0,10,1))
plt.savefig(datapath + '/Evaluations/waterfall_p12_moving_avg.svg')
plt.show()


ax = plt.subplot(111, projection='polar')
ax.plot(peak_rads_1D_12_moving_average, peak_xpos_1D_12_moving_average, linestyle = 'None',
         color='black', marker='.', markersize='2', markerfacecolor='black')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1) # clockwise
ax.grid(True)
ax.tick_params(axis='y', colors='crimson')
plt.yticks(np.arange(0,2,0.1))
plt.ylim(0,1.2)
ax.figure.set_size_inches(6,6)
plt.savefig(datapath + '/Evaluations/visual_12_alt.svg')
plt.show()

#"""
#####################################################################
#"""

#the monster. This creates the 

bm_x_12 = []
bm_y_12 = []
bitmap_scale = 300
peak_rads_1D_12_moving_average[:] = [i - ((np.pi)/2) for i in peak_rads_1D_12_moving_average]
for rad, xpos in zip(peak_rads_1D_12_moving_average, peak_xpos_1D_12_moving_average):
    x = xpos * math.sin(rad) #from polar coord to x,y
    y = xpos * math.cos(rad)
    x = ((x*bitmap_scale)/2) + (bitmap_scale/2) #Scales the actual image and centres. Alter the 2 to change size. 
    y = ((y*bitmap_scale)/2) + (bitmap_scale/2)
    bm_x_12.append(x)
    bm_y_12.append(y)
bm_x_12 = np.rint(bm_x_12)
bm_y_12 = np.rint(bm_y_12)

#removes dupes and rounds
points = np.column_stack((bm_x_12, bm_y_12))
points = np.unique(points, axis=0)
points = np.round(points).astype(int)

def rand_row(array): #finds a random row (outputs row vals. - not index)
    random_index = np.random.randint(0, len(array))
    return array[random_index]

def find_common_points(array1, array2):
    common = np.intersect1d(array1.view([('', array1.dtype)]*array1.shape[1]), 
                            array2.view([('', array2.dtype)]*array2.shape[1]))
    return common.view(array1.dtype).reshape(-1, array1.shape[1])

def local_array(point, scale): #creates array of points surrounding a given point with scale. 
    scale = int(scale)

    lup = point.copy()
    lup[0] += scale
    lup[1] += scale
    rup = point.copy()
    rup[0] += -scale
    rup[1] += scale
    lbp = point.copy()
    lbp[0] += -scale
    lbp[1] += -scale
    rbp = point.copy()
    rbp[0] += scale
    rbp[1] += -scale
    points = np.vstack((lup, rup, lbp, rbp))

    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    min_x = int(min_x)
    max_x = int(max_x)
    min_y = int(min_y)

    top = np.column_stack((np.arange(int(min_x), int(max_x) + 1), np.full(max_x - min_x + 1, min_y)))
    bottom = np.column_stack((np.arange(int(min_x), int(max_x) + 1), np.full(max_x - min_x + 1, max_y)))
    left = np.column_stack((np.full(int(max_y) - int(min_y) + 1, int(min_x)), np.arange(int(min_y), int(max_y) + 1)))
    right = np.column_stack((np.full(int(max_y) - int(min_y) + 1, int(max_x)), np.arange(int(min_y), int(max_y) + 1)))
    perimeter_points = np.vstack((top, right, bottom, left))
    perimeter_points = np.unique(perimeter_points, axis=0)
    return perimeter_points

def remove_shared_rows(array1, array2):
    mask = ~np.any(np.all(array1[:, None] == array2, axis=-1), axis=-1)

    # Return the filtered array
    return array1[mask]

def midpoints(p1, p2, scale): #produces rounded points between two points at increments
    #print('\np1:',p1, '\n& type:', type(p1))
    #print('\np2:',p2, '\n& type:', type(p2))
    x_range = np.linspace(p1[0], p2[0], scale)
    y_range = np.linspace(p1[1], p2[1], scale)
    x_range = np.round(x_range).astype(int)
    y_range = np.round(y_range).astype(int)
    points = np.column_stack((x_range, y_range))
    points = np.unique(points, axis=0)
    return points


reroll_lim = 40

points_dec = points.copy() #monitors remaining points
invisible_array = [] #invisible array with points + intepolation points. 
scale_lim = bitmap_scale/15
scale_lim = int(scale_lim)
reroll_counter = 0
obj_point_ = []
obj_point_ls = []
obj_point_np = []
common_points = []
first_pos=np.empty((0,2))
start1 = time.time()

while len(points_dec) > 0 and reroll_counter < reroll_lim:
    #reroll section
    del obj_point_, obj_point_np
    obj_point_ = rand_row(points_dec)
    obj_point_np = obj_point_.ravel()
    invisible_array.append(obj_point_np)

    move_counter = 1
    reroll = 0
    while reroll == 0 and len(points_dec) > 0 and reroll_counter < reroll_lim:
        #next point section
        i = 1
        next_pos = 0
        while next_pos == 0 and reroll == 0 and len(points_dec) > 0 and reroll_counter < reroll_lim:
            #next area scale section
            del obj_point_ls, common_points
            obj_point_ls = np.array(obj_point_np).tolist()
            perimeter_points = local_array(obj_point_ls,i)
            common_points = find_common_points(points_dec, perimeter_points)
            if len(common_points) == 0:
                i += 1
                if i > scale_lim:
                    reroll_counter += 1
                    reroll = 1
            if len(common_points) > 0:
                if i > 1:
                    if move_counter < 3:
                        first_pos = np.vstack((first_pos,obj_point_np))
                    if move_counter == 5:
                        #reintroduce the initial position
                        points_dec = np.vstack((points_dec, first_pos))
                    for p in common_points:
                        mp_x = obj_point_ls[0]
                        mp_y = obj_point_ls[1]
                        mps = midpoints([mp_x,mp_y], p, i*10)
                        invisible_array.append(mps)
                    points_dec = remove_shared_rows(points_dec, obj_point_np)
                    if len(common_points) == 1:
                        obj_point_np = common_points
                        obj_point_np = obj_point_np.ravel()
                        move_counter += 1
                    if len(common_points) > 1:
                        obj_point_np = rand_row(common_points)
                        obj_point_np = obj_point_np.ravel()
                        move_counter += 1
                    next_pos = 1
                if i == 1:
                    if move_counter == 1:
                        first_pos = obj_point_np
                    if move_counter == 5:
                        #reintroduce the initial positions
                        points_dec = np.vstack((points_dec, first_pos))
                    points_dec = remove_shared_rows(points_dec, obj_point_np)
                    if len(common_points) == 1:
                        invisible_array.append(common_points)
                        obj_point_np = common_points
                        obj_point_np = obj_point_np.ravel()
                        move_counter += 1
                    if len(common_points) > 1:
                        for pp in common_points:
                            invisible_array.append(pp)
                        obj_point_np = rand_row(common_points)
                        obj_point_np = obj_point_np.ravel()
                        move_counter += 1
                    next_pos = 1
            
invisible_array = np.vstack(invisible_array) #makes homogenous
invisible_array = np.array(invisible_array) #makes np array
invisible_array = np.unique(invisible_array, axis=0) #removes repeats

#print('\nBitmap done. Details below.\n\nReroll count:', reroll_counter,
       #'\nInvisible array count:',len(invisible_array),
       #'\nPoints remaining count:', len(points_dec), '\nPoints remaining:', points_dec,'\n')
end1 = time.time()
connect_time = end1 - start1
connect_time = str(round(connect_time, 3))
print('\nTime taken to connect:', connect_time, 's')

#"""
#####################################################################
#"""

#FILL SECTION
bitmap_centre = int(bitmap_scale/2)
brim = np.min([y for x, y in invisible_array if x == bitmap_centre])
brim_ext = brim + 5
brim_pos = np.array([brim_ext,bitmap_centre]) #inital point
brim_pos = brim_pos.astype(int)

def diamond(pos_):
    pos_u = pos_.copy()
    pos_d = pos_.copy()
    pos_l = pos_.copy()
    pos_r = pos_.copy()
    pos_u[1] += 1 
    pos_d[1] -= 1
    pos_l[0] += 1
    pos_r[0] -= 1 
    return np.vstack((pos_u, pos_d, pos_l, pos_r))

fill_array = np.array([brim_pos])
investigation_set1 = diamond(brim_pos)
investigation_set2 = np.empty((0, 2))
for d in investigation_set1:
    p = diamond(d)
    investigation_set2 = np.vstack((investigation_set2,p))
investigation_set2 = remove_shared_rows(investigation_set2, investigation_set1)
investigation_set2 = remove_shared_rows(investigation_set2, brim_pos)
investigation_set2 = np.unique(investigation_set2, axis=0)

investigation_set3 = np.empty((0, 2))
start2 = time.time()
m = 0

#reduce the second requirement if the program wont finish during this loop
while len(investigation_set1) > 0 and m<3*bitmap_scale: 

    investigation_set3 = np.empty((0, 2))
    for p in investigation_set2:
        diamond_vals = diamond(p)
        investigation_set3 = np.vstack((investigation_set3, diamond_vals))

    investigation_set3 = np.unique(investigation_set3, axis=0)
    investigation_set3 = remove_shared_rows(investigation_set3, investigation_set1)
    investigation_set3 = remove_shared_rows(investigation_set3, invisible_array)

    fill_array = np.vstack((fill_array, investigation_set1))
    fill_array = np.unique(fill_array, axis=0)
    investigation_set1 = np.empty((0,2))
    investigation_set1 = np.copy(investigation_set2)
    investigation_set2 = np.empty((0,2))
    investigation_set2 = np.copy(investigation_set3)
    
    m += 1 


end2 = time.time()
fill_time = end2 - start2
fill_time = str(round(fill_time, 3))
print('Time taken to fill:   ', fill_time, 's\n')

#"""
#####################################################################
#"""

#bitmap-specfic
bitmap = np.zeros(((bitmap_scale), (bitmap_scale), 3), dtype=np.uint8)
dots = np.linspace(0, bitmap_scale, bitmap_scale)
for i in dots:
    i = 2*int(i)
    if i < bitmap_scale:
        bitmap[i,0] = (255,255,255)
        bitmap[0,i] = (255,255,255)


x_fill = fill_array[:, 0]
y_fill = fill_array[:, 1]
for x, y in zip(x_fill, y_fill): #filled space
    x = int(x)
    y = int(y)
    bitmap[x,y] = (100, 100, 100) #RGB

    
x_invis = invisible_array[:, 0]
y_invis = invisible_array[:, 1]
for x, y in zip(x_invis, y_invis): #connecting points
    x = int(x)
    y = int(y)
    bitmap[x,y] = (180, 180, 180)

for x, y in zip(bm_x_12, bm_y_12): #points
    x = int(x)
    y = int(y)
    bitmap[x,y] = (255, 255, 255)


#NOTE THAT FOR SOME REASON, THIS BITMAP NEEDS ROTATING 90° ANTICLOCKWISE

img = Image.fromarray(bitmap, 'RGB')
img.save(datapath + '/Evaluations/bitmap_filled.png')



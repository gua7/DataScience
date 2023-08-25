# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 19:49:24 2023

@author: Jiaqi
"""
import numpy as np
import netCDF4 as nc
import xarray as xr
import numpy.ma as ma
from clusterpy import cluster
import xarray as xr
import os
import datetime
import pytz

def mycluster(field,threshold,large_equal):
    # package from https://github.com/lochbika/clusterpy
    # First install this pakage use setup.py
    # It requires to create a temp file... Not an optimal cluster algorithm but fine
    file_path = 'temp.nc'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted and recreated")  
    ds = nc.Dataset('temp.nc','w', format='NETCDF4')
    ds.createDimension('x', len(field[:,0]))
    ds.createDimension('y', len(field[0,:]))
    ds.createDimension('time', len(np.zeros((1))))
    ds.createVariable("x","f",("x",))
    ds.createVariable("y","f",("y",))
    ds.createVariable("RR","f",("time","x","y",))
    ds.close() 
    ds = nc.Dataset('temp.nc',mode='a')
    ds['RR'][0,:,:]= field
    ds.close()   
    infile='temp.nc'
    thres = threshold
    ds_disk = xr.open_dataset(infile, decode_times=False)
    tslice = ds_disk.isel(time=0, x=range(ds_disk.sizes['x']), y=range(ds_disk.sizes['y'])).to_array()
    if large_equal:
        cldata = xr.where(tslice > -999999999, -1, -1)
        cells = cluster.ClusterArray(tslice,thres).get_clusterarray()
        cells = cells[0,:,:] 
        ds_disk.close()
    else:
        cldata = xr.where(tslice > -999999999, -1, -1)
        cells = cluster.ClusterArray(-tslice,-thres).get_clusterarray()
        cells = cells[0,:,:] 
        ds_disk.close()  
    os.remove(file_path)
    return cells


def InversedDistance(input_arr): #Inversed distance interpolation
    mask_idx = np.where(input_arr.mask)[0] #Find where needs to be interpolated
    for idx in mask_idx:
        unmasked_idx = np.where(~input_arr.mask) #Find where is the data
        distances = np.abs(unmasked_idx - idx) #calculate distances
        inverse_distances = 1.0 / distances
        normalized_inverse_distances = inverse_distances / np.sum(inverse_distances)
        input_arr[idx] = np.sum(input_arr[unmasked_idx] * normalized_inverse_distances) #fill the interpolated
    return input_arr
        

def ImprovedDetecting(Z,vel,ML): #original ML
    classification = np.zeros_like(Z)
    Z = ma.masked_where(vel>6/20*Z + 12, Z) #mask clutter
    melting_level = np.zeros_like(ML[:,0])  #creating melting_level for each time
    
    for i in range(len(melting_level)): # find where is the used data
        if ma.all(ML[i, :]) or np.nanmax(ML[i, :]) < 0.6: 
            melting_level[i] = ma.masked #mask the value if it is smaller than 0.6 or not presented
        else:
            melting_level[i] = np.nanargmax(ML[i, :])
    cell_melting = InversedDistance(melting_level).astype(int) #interpolation, if all masked it returns all masked
    field_cell = mycluster(Z,threshold=99999,large_equal=False) #get cells
    field_cell = ma.masked_less(field_cell.data,0) 
    max_field_cell = np.nanmax(field_cell)+1
    
    for i in range(int(max_field_cell)):   #mask small cells
        time_idx, range_idx = np.where(field_cell == i)
        first_column = np.min(time_idx)
        last_column = np.max(time_idx)
        if last_column - first_column < 60/10: #mask cells last less than 1 min, and continue
            field_cell = ma.masked_equal(field_cell,i)
            continue  
        if ma.any(~melting_level[first_column:last_column+1].mask): #mask melting layer
            for t in range(last_column-first_column+1):
                melting_idx = np.where((field_cell[t + first_column, :]==i) & (np.arange(len(field_cell[t + first_column, :])) > cell_melting[t + first_column]))[0]
                classification[t+first_column,melting_idx]=1
                field_cell[t+first_column,melting_idx]=ma.masked  
                
    for i in range(len(field_cell[:,0])): #detecting virga based on velocity
        if ma.any(~field_cell.mask):
            max_field_cell2 = np.nanmax(field_cell)+1
            for j in range(int(max_field_cell2)):
                cell_range_idx = np.where(field_cell[i,:]==j)[0]
                if cell_range_idx.size > 0:
                   max_vel = 1/600*35*np.min(cell_range_idx)+1.5
                   if vel[i,np.min(cell_range_idx)]> max_vel or np.min(cell_range_idx)<10:
                       classification[i,cell_range_idx]=3    
                   else:
                       classification[i,cell_range_idx]=2 

    masked_virga = ma.masked_where(classification != 2, classification) #Filter out small virga cells and assign 'Undefined'
    virga_cells = mycluster(masked_virga,threshold=99999,large_equal=False)
    virga_cells = ma.masked_less(virga_cells.data,0)  
    if ma.any(~virga_cells.mask):
        max_virga_cell = np.nanmax(virga_cells)+1
        for j in range(int(max_virga_cell)):  #mask cells last less than 1 min
            t_idx, r_idx = np.where(virga_cells == j)
            first_col = np.min(t_idx)
            last_col = np.max(t_idx)
            if last_col-first_col< 6:
                classification[np.where(virga_cells==j)]=4
                virga_cells = ma.masked_equal(virga_cells,j)
                             
    masked_prec = ma.masked_where(classification != 3, classification) #Filter out small virga cells and assign 'Undefined'
    prec_cells = mycluster(masked_prec,threshold=99999,large_equal=False)
    prec_cells = ma.masked_less(prec_cells.data,0)
    if ma.any(~prec_cells.mask):
       max_prec_cell = np.nanmax(prec_cells)+1
       for j in range(int(max_prec_cell)):  #mask cells last less than 1 min
            t_idx, r_idx = np.where(prec_cells == j)
            first_col = np.min(t_idx)
            last_col = np.max(t_idx)
            if last_col-first_col< 6: 
                classification[np.where(prec_cells==j)]=4
                prec_cells = ma.masked_equal(prec_cells,j)
                
    return classification,virga_cells,prec_cells

def ImprovedDetecting2(Z,vel,ML): 
    #Here I tried another methods, which is 'eveything, either virga or surface precipitation
    #must fall above 300 m, and I set a lower height threshold to distinguish virga
    #and surface precipitation, the results are OK, but sometimes melting layer is below 300m
    #which may cause the problem.
    classification = np.zeros_like(Z)
    Z = ma.masked_where(vel>6/20*Z + 12, Z) #mask clutter
    melting_level = np.zeros_like(ML[:,0])  #creating melting_level for each time
    
    for i in range(len(melting_level)):
        if ma.all(ML[i, :]) or np.nanmax(ML[i, :]) < 0.6:
            melting_level[i] = ma.masked
        else:
            melting_level[i] = np.nanargmax(ML[i, :])
            
    cell_melting = InversedDistance(melting_level).astype(int) 
    field_cell = mycluster(Z,threshold=99999,large_equal=False) #get cells
    field_cell = ma.masked_less(field_cell.data,0)
    max_field_cell = np.nanmax(field_cell)+1
    for i in range(int(max_field_cell)):   #mask small cells
        time_idx, range_idx = np.where(field_cell == i)
        first_column = np.min(time_idx)
        last_column = np.max(time_idx)
        if last_column - first_column < 60/10: #mask cells last less than 1 min, and continue
            field_cell = ma.masked_equal(field_cell,i)
            continue  
        if ma.any(~melting_level[first_column:last_column+1].mask): #mask melting layer
            for t in range(last_column-first_column+1):
                melting_idx = np.where((field_cell[t + first_column, :]==i) & (np.arange(len(field_cell[t + first_column, :])) > cell_melting[t + first_column]))[0]
                classification[t+first_column,melting_idx]=1
                field_cell[t+first_column,melting_idx]=ma.masked  
    for i in range(len(field_cell[:,0])): #detecting virga based on velocity
        if ma.any(~field_cell.mask):
            max_field_cell2 = np.nanmax(field_cell)+1
            for j in range(int(max_field_cell2)):
                cell_range_idx = np.where(field_cell[i,:]==j)[0]
                if cell_range_idx.size > 0:
                   max_vel = 1/600*35*np.min(cell_range_idx)+1.5
                   if (vel[i, np.min(cell_range_idx)] <= max_vel and np.min(cell_range_idx) >= 4) and np.max(cell_range_idx) >= 10:
                       classification[i,cell_range_idx]=2    
                
                   elif (vel[i, np.min(cell_range_idx)] > max_vel or np.min(cell_range_idx) < 4) and np.max(cell_range_idx) >= 10:
                       classification[i,cell_range_idx]=3
                   else:
                       classification[i,cell_range_idx]=4 
    masked_virga = ma.masked_where(classification != 2, classification)
    virga_cells = mycluster(masked_virga,threshold=99999,large_equal=False)
    virga_cells = ma.masked_less(virga_cells.data,0)
    if ma.any(~virga_cells.mask):
        max_virga_cell = np.nanmax(virga_cells)+1
        for j in range(int(max_virga_cell)):  #mask cells last less than 1 min
            t_idx, r_idx = np.where(virga_cells == j)
            first_col = np.min(t_idx)
            last_col = np.max(t_idx)
            if last_col-first_col< 6:
                classification[np.where(virga_cells==j)]=4
                virga_cells = ma.masked_equal(virga_cells,j)
    masked_prec = ma.masked_where(classification != 3, classification)
    prec_cells = mycluster(masked_prec,threshold=99999,large_equal=False)
    prec_cells = ma.masked_less(prec_cells.data,0)
    if ma.any(~prec_cells.mask):
       max_prec_cell = np.nanmax(prec_cells)+1
       for j in range(int(max_prec_cell)):  #mask cells last less than 1 min
            t_idx, r_idx = np.where(prec_cells == j)
            first_col = np.min(t_idx)
            last_col = np.max(t_idx)
            if last_col-first_col< 6: 
                classification[np.where(prec_cells==j)]=4
                prec_cells = ma.masked_equal(prec_cells,j)
    return classification,virga_cells,prec_cells


def merge_cells(cells, gap=5): #Here 1 min and 200 m are 6 columns and rows
    if ma.any(~cells.mask):
        for label_id in range(0, int(np.max(cells)) + 1):
            cell_indices = np.where(cells == label_id)
            if cell_indices[0].size > 0:
                min_row = np.min(cell_indices[0])
                max_row = np.max(cell_indices[0])
                min_col = np.min(cell_indices[1])
                max_col = np.max(cell_indices[1])
                for other_label_id in range(0,int(np.max(cells)) + 1): #see if other cells are close enough
                    if label_id != other_label_id:  
                        other_indices = np.where(cells == other_label_id)
                        if other_indices[0].size > 0:
                            other_min_row = np.min(other_indices[0])
                            other_max_row = np.max(other_indices[0])
                            other_min_col = np.min(other_indices[1])
                            other_max_col = np.max(other_indices[1])          
                            row_diff = min(abs(min_row - other_min_row), abs(min_row - other_max_row),
                                           abs(max_row - other_min_row), abs(max_row - other_max_row))
                            col_diff = min(abs(min_col - other_min_col), abs(min_col - other_max_col),
                                           abs(max_col - other_min_col), abs(max_col - other_max_col))
                            if row_diff <= gap and col_diff <= gap:
                                cells[cell_indices] = label_id
                                cells[other_indices] = label_id                            
    return cells 

def last_sunday_in_month(year, month): #Find Dutch summer time and winter time.
    last_day = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    offset = (last_day.weekday() - 6) % 7 
    return last_day - datetime.timedelta(days=offset)

def time_convert(epoch_times,data):
    # Here I did the mannual correction, but the better one is in time_convert_noshift
    
    
    # data = ma.filled(data,0)
    # timestamps = [datetime.datetime.fromtimestamp(epoch_time)- datetime.timedelta(hours=2) for epoch_time in epoch_times]
    timestamps = []
    for epoch_time in epoch_times:
        # Convert epoch time to a datetime object
        timestamp = datetime.datetime.fromtimestamp(epoch_time)
        # Determine the start and end dates of DST for the given year
        dst_start = last_sunday_in_month(timestamp.year, 3)
        dst_end = last_sunday_in_month(timestamp.year, 10)
        dst_start = datetime.datetime.combine(dst_start, datetime.time.min)
        dst_end = datetime.datetime.combine(dst_end, datetime.time.min)
        # Adjust the timedelta based on whether the timestamp is within DST
        if dst_start <= timestamp < dst_end:
            # Within DST period, subtract 2 hours
            adjusted_timestamp = timestamp - datetime.timedelta(hours=2)
        else:
            # Outside DST period, subtract 1 hour
            adjusted_timestamp = timestamp - datetime.timedelta(hours=1)
        timestamps.append(adjusted_timestamp)  
    start_date = timestamps[0].date()
    end_date = timestamps[-1].date()
    # Initialize the dictionary to store data by dates
    data_by_dates = {}
    # Iterate over dates
    current_date = start_date
    while current_date <= end_date:
        # Get the start and end datetime objects for the current date
        start_datetime = datetime.datetime.combine(current_date, datetime.time.min)
        end_datetime = datetime.datetime.combine(current_date, datetime.time.max)
        # Filter the timestamps for the current date
        date_timestamps = [ts for ts in timestamps if start_datetime <= ts <= end_datetime]
        # Calculate the number of desired points for the current date
        desired_points = 8640
        # Create an array of timestamps with 10-second intervals
        time_range = np.arange(start_datetime, end_datetime, datetime.timedelta(seconds=10)).astype(datetime.datetime)
        # Create an array to store the data for the current date
        date_data = ma.masked_array(np.zeros(desired_points))
        # Fill in the data array with the closest available values
        if len(date_timestamps) > 0:
            for i, ts in enumerate(time_range):
                closest_timestamp = min(date_timestamps, key=lambda x: abs(x - ts))
                closest_index = timestamps.index(closest_timestamp)
                time_diff = abs(ts - closest_timestamp).total_seconds()
                if time_diff <= 5:
                    date_data[i] = data[closest_index]
                else:
                    date_data[i] = ma.masked
        data_by_dates[current_date] = date_data
        print(current_date)
        current_date += datetime.timedelta(days=1)
    return data_by_dates
        
        
               
def time_convert_noshift(epoch_times, data):
    # data = ma.filled(data,0)
    timestamps = [datetime.datetime.fromtimestamp(epoch_time, pytz.UTC) for epoch_time in epoch_times]
    start_date = timestamps[0].date()
    end_date = timestamps[-1].date()
    # Initialize the dictionary to store data by dates
    data_by_dates = {}
    # Iterate over dates
    current_date = start_date
    while current_date <= end_date:
        # Get the start and end datetime objects for the current date (in UTC)
        start_datetime = datetime.datetime.combine(current_date, datetime.time.min, tzinfo=pytz.UTC)
        end_datetime = datetime.datetime.combine(current_date, datetime.time.max, tzinfo=pytz.UTC)
        # Filter the timestamps for the current date
        date_timestamps = [ts for ts in timestamps if start_datetime <= ts <= end_datetime]
        # Calculate the number of desired points for the current date
        desired_points = 8640
        # Create an array of timestamps with 10-second intervals (in UTC)
        time_range = np.arange(start_datetime, end_datetime, datetime.timedelta(seconds=10))
        time_range = np.array([ts.item().replace(tzinfo=pytz.UTC) for ts in time_range])
        # Create an array to store the data for the current date
        date_data = ma.masked_array(np.zeros(desired_points))
        # Fill in the data array with the closest available values
        if len(date_timestamps) > 0:
            for i, ts in enumerate(time_range):
                closest_timestamp = min(date_timestamps, key=lambda x: abs(x - ts))
                closest_index = timestamps.index(closest_timestamp)
                time_diff = abs(ts - closest_timestamp).total_seconds()
                if time_diff <= 5:
                    date_data[i] = data[closest_index]
                else:
                    date_data[i] = ma.masked
        data_by_dates[current_date] = date_data
        print(current_date)
        current_date += datetime.timedelta(days=1)
    return data_by_dates
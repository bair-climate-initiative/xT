def compute_iter_per_batch(start_year_train,end_year_train,hours_per_year,batch_size,num_devices,predict_range,lookback_range,n_chunk,hrs_each_step=1):
    global_bs = batch_size * num_devices
    total_samples = (end_year_train - start_year_train + 1) * (hours_per_year - (predict_range+lookback_range)*n_chunk) / hrs_each_step
    return int(total_samples // global_bs + 1)
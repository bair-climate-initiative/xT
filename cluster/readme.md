## Run jobs locally

e.g.
```
/cluster/main_exp_local.sh  ~/xView3_second_place/cluster/exps/512/swin_t_bs_4_p0.5.sh
```

where the first argument is the actual config file

## Run jobs on cluster

e.g.
```
/cluster/main_exp_cluster.sh  ~/xView3_detection/cluster/exps/512/swin_t_bs_4_p0.5.sh
```

where the first argument is the actual config file

## Run everything in folder
```
./cluster/main_exp_all.sh ~/xView3_second_place/cluster/main_exp_local.sh
``` 
or 

```
./cluster/main_exp_all.sh ~/xView3_detection/cluster/main_exp_cluster.sh
```
This will loop over all files in exps folder
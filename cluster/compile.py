import argparse
import os
def get_str(fpath):
    with open(fpath) as f:
        data = f.read()
    return data

def compile(template,cfg_sh):
    args = (
        get_str("base_project_config.sh"),
        get_str("cluster_housekeeping.sh"),
        get_str(cfg_sh),
        get_str("run.sh")
    )
    return template.format(*args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,default="exps/512/swin_l_bs_4_lr_3e-5_ep60.sh")
    parser.add_argument('-t','--template',type=str,default="main_exp_cluster.template")
    parser.add_argument('-o','--output',type=str,default="")
    args = parser.parse_args()
    TEMPLATE = get_str(args.template)
    r = compile(TEMPLATE,args.config)
    if args.output:
        with open(os.path.join('build',args.output.replace('/','_').replace('\\','_')),'w') as f:
            f.write(r)
    else:
        print(r)
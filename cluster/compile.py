import argparse
import os
def get_str(fpath):
    with open(fpath) as f:
        data = f.read()
    return data

def read_config(cfg_sh):
    lines = get_str(cfg_sh)
    lines = lines.split('\n')
    out = {}
    for r in lines:
       args = r.split('=')
       if len(args)==2:
           k,v = args
           if v[0] == '"' and v[-1] ==v[0]:
               v = v[1:-1]
           out[k]=v
    return out       
def compile(template,cfg_sh):
    cfgs = read_config("base_project_config.sh")
    cfgs.update(read_config(cfg_sh))
    cfgs['DEVICES']=','.join([str(x) for x in range(int(cfgs['NUM_GPUS']))])
    cmd = get_str("run_cluster.sh")
    items = list(cfgs.items())
    items = sorted(items,key=lambda x:-len(x[0]) )
    for k,v in items:
        cmd = cmd.replace(f'${k}',v)
    return cmd,cfgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,default="exps/512/swin_l_bs_4_lr_3e-5_ep60.sh")
    parser.add_argument('-t','--template',type=str,default="main_exp_cluster.template")
    parser.add_argument('-o','--output',type=str,default="")
    args = parser.parse_args()
    TEMPLATE = get_str(args.template)
    cmd,cfgs = compile(TEMPLATE,args.config)
    if args.output:
        out_dir = os.path.join('build',
                               f"gpu_{cfgs['REQUIRED_VRAM']}X{cfgs['NUM_GPUS']}")
        os.makedirs(out_dir,exist_ok=True)
        with open(os.path.join(out_dir,args.output.replace('/','_').replace('\\','_')),'w') as f:
            f.write(cmd)
    else:
        print(cmd)
import wandb
import pandas as pd
api = wandb.Api()
# wandb_args = dict(
#     project="xview3 detection unet",
#     entity="bair-climate-initiative",
#     resume="allow",
#     name=train_config.name,
#     config=self.conf,
# )
runs = api.runs(path="bair-climate-initiative/xview3 detection unet")
for r in runs:
    summary = r.summary
    if 'loc_fscore_shore' in summary:
        df = pd.DataFrame(r.scan_history(keys=['loc_fscore_shore']))
        max_val = df['loc_fscore_shore'].max()
        summary['best_shore'] = max_val
        try:
            summary.update()
            print("SUCCESS")
        except:
            pass
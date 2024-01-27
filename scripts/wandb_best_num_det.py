import pandas as pd
from tqdm.cli import tqdm
from joblib import Parallel
import wandb

api = wandb.Api(api_key="553070c5ef0d454bcb1e91afaabf2359ef69f4a0")
# wandb_args = dict(
#     project="xview3 detection unet",
#     entity="bair-climate-initiative",
#     resume="allow",
#     name=train_config.name,
#     config=self.conf,
# )
runs = api.runs(path="bair-climate-initiative/Vit-XL-mmdetection")
for r in tqdm(runs):
    summary = r.summary
    # if "loc_fscore_shore" in summary:
    #     df = pd.DataFrame(r.scan_history(keys=["loc_fscore_shore"]))
    #     try:
    #         max_val = df["loc_fscore_shore"].max()
    #     except:
    #         max_val = 0
    #     summary["best_shore"] = max_val
    #     try:
    #         summary.update()
    #         # print("SUCCESS")
    #     except:
    #         print("FAIL")
    
    for k in [
        'coco/bbox_mAP','epoch','coco/bbox_mAP_s'
    ]:
        try:
            df = pd.DataFrame(r.scan_history(keys=[k]))
            try:
                max_val = df[k].max()
            except:
                max_val = 0
            summary["best__"+k] = max_val
            try:
                summary.update()
                # print("SUCCESS")
            except:
                print("FAIL")
        except:
            pass

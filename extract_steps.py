import os
import torch
import json
from pathlib import Path
from typing import List, Dict

class Save_Steps:

    def __init__(self,output_dir="/Users/muhammadelmahdi/Prof_Ommer-Lab-Task/project/data/diffusion_steps"):
        self.output_dir=Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
    

    


    def save_steps(self,steps):
        data=[]
        for step in steps:
            s_number=step['step']
            time_stamp=step['timestep']
            step_dir=self.output_dir/f"step_{s_number}"
            step_dir.mkdir(exist_ok=True)
            torch.save(step["latent"], step_dir / "latent.pt")


            torch.save(step["noise_pred"], step_dir / "noise_pred.pt")
            data_entry = {
                "step": s_number,
                "timestep": time_stamp,
                "latent_path": str(step_dir / "latent.pt"),
                "noise_pred_path": str(step_dir / "noise_pred.pt")
            }
            data.append(data_entry)

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(data, f, indent=4)

    

        








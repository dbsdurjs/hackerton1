import os
from diffusers import StableDiffusionPipeline
import torch, time
import matplotlib.pylab as plt

def sampling_func():
    unique_identifiers = ["sks", "zwx", "swz", "kql", "ifg", "wpb"]
    num_images = 1800
    num_model = 6

    for k in range(num_model):
        pipeline = StableDiffusionPipeline.from_pretrained(f"yoon6173/result_soldier{k+1}", torch_dtype=torch.float16).to("cuda")
        unique_identifier = unique_identifiers[k]
        
        prompt_list = [ f"photo of camouflaged {unique_identifier} soldier hiding the large tree in forest", # 1
                        # f"Photo of camouflaged {unique_identifier} soldier behind the brown bushes with wide distance viewpoint, blending naturally with the background.", # 2
                        # f"Photo of camouflaged {unique_identifier} soldier crouching low within various textures of leaves, branches, and shadows, naturally concealed, with wide distance viewpoint", # 3
            ]
        
        print(f'{k+1}번째 model sampling, unique identifier : {unique_identifiers[k]}')
        for i in range(len(prompt_list)):
            prompt = prompt_list[i]

            for j in range(num_images):
                image = pipeline(prompt, num_inference_steps=50, guidance_sacle=7.5, seed=time.time()).images[0]

                image_path = f'../sampling_camouflage_soldier_exp_temp_(1)/soldier_{k + 1}_{(i * num_images) + (j + 1)}.png'   #change
                image.save(image_path)
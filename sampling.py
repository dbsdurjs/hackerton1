import os
from diffusers import StableDiffusionPipeline
import torch, time
import matplotlib.pylab as plt

def sampling_func():
    unique_identifiers = ["sks", "zwx", "swz", "kql", "ifg", "wpb"]
    num_images = 20
    num_model = 6

    for k in range(num_model):
        pipeline = StableDiffusionPipeline.from_pretrained(f"yoon6173/result_soldier{k+1}", torch_dtype=torch.float16).to("cuda")
        unique_identifier = unique_identifiers[k]
        
        prompt_list = [ f"photo of camouflaged {unique_identifier} soldier hiding the large tree in forest", # 1
                        f"Photo of camouflaged {unique_identifier} soldier sitting quietly, almost indistinguishable from the surroundings in a dense forest scene.",
                        f"Photo of camouflaged {unique_identifier} soldier crouched subtly among dense undergrowth and tall trees in an expansive woodland setting.",
                        f"Photo of camouflaged {unique_identifier} soldier lying low, viewed from above for a bird’s-eye perspective, naturally blending into the environment.",
                        f"Photo of camouflaged {unique_identifier} soldier behind the brown bushes with wide distance viewpoint, blending naturally with the background.", # 2
                        f"Photo of  camouflaged {unique_identifier} soldier hiding motionless with wide distance viewpoint in a dense, shadowy forest, blending seamlessly and naturally into the environment, shot from a slightly elevated angle.",
                        f"Photo of camouflaged {unique_identifier} soldier crouching low within various textures of leaves, branches, and shadows, naturally concealed, with wide distance viewpoint", # 3
                        f"Photo of camouflaged {unique_identifier} soldier lying prone in a serene woodland, blending naturally into the background, captured from a distant side angle with wide distance viewpoint.",
                        f"Photo of camouflaged {unique_identifier} soldier hidden filled with trees, bushes, and natural elements, nearly undetectable while lying flat, blending naturally with the surroundings and wide distance viewpoint.",
                        f"Photo of camouflaged {unique_identifier} soldier in a forest environment filled with intricate natural details, viewed from a wide perspective.",
                        f"Photo of camouflaged {unique_identifier} soldier, with natural sunlight filtering through, taken from a long angle.",
                        f"photo of camouflaged {unique_identifier} soldier with full tactical gear and equiment, viewed from a long perspective",
                        f"photo of camouflaged {unique_identifier} soldier observing target through high-powered scope, viewed from a long perspective",
                        f"photo of camouflaged {unique_identifier} soldier coordinating with team members using hand signals, viewed from a long perspective",
                        f"Photo of camouflaged {unique_identifier} soldier with team members preparing defensive position using natural camouflage materials, viewed from a long perspective",
            ]
        
        print(f'{k+1}번째 model sampling, unique identifier : {unique_identifiers[k]}')
        for i in range(len(prompt_list)):
            prompt = prompt_list[i]

            for j in range(num_images):
                image = pipeline(prompt, num_inference_steps=50, guidance_sacle=7.5, seed=time.time()).images[0]

                image_path = f'./sampling_camouflage_soldier_exp/soldier_{k + 1}_{(i * num_images) + (j + 1)}.png'   #change
                image.save(image_path)
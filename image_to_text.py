import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import torch
from util import mkdir
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_captions_batch(image_paths):
    images = [Image.open(path).convert('RGB') for path in image_paths]
    inputs = processor(images, return_tensors="pt", padding=True).to(device)
    out = model.generate(**inputs)
    captions = [processor.decode(o, skip_special_tokens=True) for o in out]
    return captions


dataroots = "/home/HDD/yjz/dataset/AIGIBench"
dataroots_text = "/home/HDD/yjz/dataset/AIGIBench_text"
mkdir(dataroots_text)

for v_id, val in enumerate(os.listdir(dataroots)):
    image_paths = []
    captions_dict = {}

    dataroot = '{}/{}'.format(dataroots, val)
    classes = os.listdir(dataroot)

    for cls in classes:
        if '0_real' not in classes or '1_fake' not in classes:
            for subfolder in ['0_real', '1_fake']:
                folder_path = os.path.join(dataroot, cls, subfolder)
                if os.path.exists(folder_path):
                    for img_name in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, img_name)
                        image_paths.append(img_path)
        else:
            folder_path = os.path.join(dataroot, cls)
            if os.path.exists(folder_path):
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    image_paths.append(img_path)

    batch_size = 64
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        captions = generate_captions_batch(batch_paths)
        for path, caption in zip(batch_paths, captions):
            captions_dict[path] = caption
            print(path + '\t: ' + caption)

    with open('{}/{}.json'.format(dataroots_text, val), 'w', encoding='utf-8') as f:
        json.dump(captions_dict, f, ensure_ascii=False, indent=4)


workdir = "/home/fernando/Documents/home-pipelines-2/vision/object_detector/"

from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from segment_anything import sam_model_registry, SamPredictor
import argparse
import imutils
import time
import ultralytics
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import csv
import yaml
from pycocotools import mask
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, UnidentifiedImageError
import cv2
import numpy as np
import json
import random
import sys
import os
os.chdir("/")

async def crop():
    from PIL import Image, UnidentifiedImageError, ImageOps
    import os
    
    base_path = workdir + "images"
    
    for class_dir in os.listdir(base_path):
        class_path = os.path.join(base_path, class_dir)
    
        if not os.path.isdir(class_path):
            continue
        print(class_path)
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
    
            try:
                with Image.open(file_path) as img:
                    # Apply EXIF orientation
                    img = ImageOps.exif_transpose(img)
    
                    width, height = img.size
                    min_dim = min(width, height)
    
                    left = (width - min_dim) // 2
                    top = (height - min_dim) // 2
                    right = left + min_dim
                    bottom = top + min_dim
    
                    img_cropped = img.crop((left, top, right, bottom))
                    img_cropped.save(file_path)
            except (UnidentifiedImageError, OSError) as e:
                print(f"Removing corrupt file: {file_path}")
                os.remove(file_path)
    
    os.chdir(workdir)


async def resize():
    base_path = workdir + "images"
    size = 1280
    
    for class_dir in os.listdir(base_path):
        class_path = os.path.join(base_path, class_dir)
    
        if not os.path.isdir(class_path):
            continue
        print(class_path)
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
    
            try:
                with Image.open(file_path) as img:
                    img = img.resize((size, size))
                    img.save(file_path)
            except (UnidentifiedImageError, OSError) as e:
                print(f"Removing corrupt file: {file_path}")
                os.remove(file_path)  # delete corrupt image
    
    os.chdir(workdir)


async def segment():
    SAVE_BB = False
    DEBUG = False
    
    # path to save results already processed and segmented images
    results_path = workdir + "processed"
    # change the path of the model config file
    config_file = workdir + "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    # change the path of the model
    checkpoint_path = workdir + "groundingdino_swint_ogc.pth"
    output_dir = results_path
    box_threshold = 0.3
    text_threshold = 0.25
    token_spans = None
    
    sys.path.append("..")
    sam_model = "h"
    
    # use sam model
    # wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    if sam_model == "h":
        sam_checkpoint = workdir + "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
    else:
        sam_checkpoint = workdir + "sam_vit_l_0b3195.pth"
        model_type = "vit_l"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    images = []
    annotations = []
    categories = []
    
    img_id = 0
    anno_id = 0
    
    # Make a list of all the directories in the path
    base_path = workdir + "images"
    path_to_classes = [f.path for f in os.scandir(base_path) if f.is_dir()]
    
    
    def load_image(image_path):
    
        image_pil = Image.open(image_path).convert("RGB")  # load image
    
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image
    
    
    def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False)
        if DEBUG:
            print(load_res)
        _ = model.eval()
        return model
    
    
    def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
        assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            if DEBUG:
                print("Running model...")
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
    
        # filter output
        if token_spans is None:
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    
            # get phrase
            tokenlizer = model.tokenizer
            tokenized = tokenlizer(caption)
            # build pred
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(
                    logit > text_threshold, tokenized, tokenlizer)
                if with_logits:
                    pred_phrases.append(
                        pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)
        else:
            # given-phrase mode
            positive_maps = create_positive_map_from_span(
                model.tokenizer(text_prompt),
                token_span=token_spans
            ).to(image.device)  # n_phrase, 256
    
            logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
                # get phrase
                phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                # get mask
                filt_mask = logit_phr > box_threshold
                # filt box
                all_boxes.append(boxes[filt_mask])
                # filt logits
                all_logits.append(logit_phr[filt_mask])
                if with_logits:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend(
                        [phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases
        return boxes_filt, pred_phrases
    
    
    def verify_or_create_dir(path):
        os.makedirs(path, exist_ok=True)
        if DEBUG:
            print(f"Verified/created: {path}")
    
    
    def count_all_files_in_dir(directory):
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len([f for f in files if os.path.isfile(os.path.join(root, f))])
        return count
    
    
    # Get total number of images to process
    image_dir = workdir + "images"
    number_of_images = count_all_files_in_dir(image_dir)
    print(f"Total image files: {number_of_images}")
    
    # Check if results directory exists, else create it
    verify_or_create_dir(results_path)
    
    # Main loop
    i = 0
    for class_path in path_to_classes:
        imgPaths = os.listdir(class_path)
        if SAVE_BB:
            class_name = os.path.basename(class_path)
            verify_or_create_dir(f"{results_path}/bbs/{class_name}")
    
        for imgPath in imgPaths:
            if DEBUG:
                print(f"Processing image: {imgPath}")
            print(f"%{i * 100 / number_of_images}")
            img = imutils.resize(cv2.imread(f"{class_path}/{imgPath}"))
            if img is None:
                continue
    
        # ------------------------start grounding----------------------------------------------
    
            # Image_path = args.image_path
            cpu_only = False if torch.cuda.is_available() else True
    
            # Load image
            image_pil, image = load_image(f"{class_path}/{imgPath}")
    
            # Load model
            model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)
    
            # Set the text_threshold to None if token_spans is set.
            if token_spans is not None:
                text_threshold = None
                print("Using token_spans. Set the text_threshold to None.")
    
            # Run model
            text_prompt = os.path.basename(class_path)
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, cpu_only=cpu_only, token_spans=eval(
                    f"{token_spans}")
            )
    
            # Found bb dimensions
    
            size = image_pil.size
            pred_dict = {
                "boxes": boxes_filt,
                "size": [size[1], size[0]],  # H, W
                "labels": pred_phrases,
            }
    
            H, W = pred_dict["size"]
            boxes = pred_dict["boxes"]
            labels = pred_dict["labels"]
            assert len(boxes) == len(
                labels), "boxes and labels must have same length"
    
            draw = ImageDraw.Draw(image_pil)
            mask = Image.new("L", image_pil.size, 0)
            mask_draw = ImageDraw.Draw(mask)
    
            # change pil image to cv2 image
            img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            img2 = img.copy()
            # draw boxes and masks
            for box, label in zip(boxes, labels):
                # from 0..1 to 0..W, 0..H
                box = box * torch.Tensor([W, H, W, H])
                # from xywh to xyxy
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]
                # random color
                color = tuple(np.random.randint(0, 255, size=1).tolist())
                # draw
                padding = 10
                x0, y0, x1, y1 = box
                x0, y0, x1, y1 = int(x0)-padding, int(y0) - \
                    padding, int(x1)+padding, int(y1)+padding
    
                # validate if the bounding box is inside the image
                if x0 < 0:
                    x0 = 0
                if y0 < 0:
                    y0 = 0
                if x1 > W:
                    x1 = W
                if y1 > H:
                    y1 = H
    
                # draw rectangles
                cv2.rectangle(img2, (x0, y0), (x1, y1), color, 2)
    
                draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
                # draw.text((x0, y0), str(label), fill=color)
    
                font = ImageFont.load_default()
                if hasattr(font, "getbbox"):
                    bbox = draw.textbbox((x0, y0), str(label), font)
                else:
                    w, h = draw.textsize(str(label), font)
                    bbox = (x0, y0, w + x0, y0 + h)
                # bbox = draw.textbbox((x0, y0), str(label))
                draw.rectangle(bbox, fill=color)
                draw.text((x0, y0), str(label), fill="white")
    
                mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
    
        # ----------------Start SAM--------------------------------------------------------------
    
                class_name = class_path.split("/")[-1]
                sam_bounding_box = np.array([x0, y0, x1, y1])
                ran_sam = False
                # run sam
                if ran_sam == False:
                    predictor.set_image(img)
                    ran_sam = True
    
                mask, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=sam_bounding_box,
                    multimask_output=False,
                )
    
                mask, _, _ = predictor.predict(
                    box=sam_bounding_box, multimask_output=False)
    
                # Make png mask
                contours, _ = cv2.findContours(mask[0].astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Your call to find the contours
    
                # threshold input image using otsu thresholding as mask and refine with morphology
                ret, pngmask = cv2.threshold(mask[0].astype(
                    np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                kernel = np.ones((9, 9), np.uint8)
                pngmask = cv2.morphologyEx(pngmask, cv2.MORPH_CLOSE, kernel)
                pngmask = cv2.morphologyEx(pngmask, cv2.MORPH_OPEN, kernel)
                result = img.copy()
                result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
                result[:, :, 3] = pngmask
    
        # ----------------Save Images-----------------------------------------------------------------
    
                if SAVE_BB:
                    cv2.imwrite(f"{results_path}/bbs/{class_name}/{imgPath}", img2)
    
                verify_or_create_dir(f"{results_path}/{class_name}")
    
                file_path = f"{results_path}/{class_name}/{imgPath[:-4]}.png"
                if os.path.exists(file_path):
                    if os.path.exists(f"{results_path}/{class_name}/{imgPath[:-4]}_1.png"):
                        if DEBUG:
                            print("File already exists, saving with _2")
                        cv2.imwrite(
                            f"{results_path}/{class_name}/{imgPath[:-4]}_2.png", result)
                    else:
                        if DEBUG:
                            print("File already exists, saving with _1")
                        file_path = f"{results_path}/{class_name}/{imgPath[:-4]}_1.png"
    
                cv2.imwrite(file_path, result)
                ran_sam = False
            i = i + 1


async def crop_precessed():
    from PIL import Image
    
    
    def verify_or_create_dir(path):
        os.makedirs(path, exist_ok=True)
    
    
    results_path = workdir + "DS_res/"
    path_to_classes = [f.path for f in os.scandir(
        workdir + "processed") if f.is_dir()]
    
    for class_path in path_to_classes:
        class_name = os.path.basename(class_path)
        verify_or_create_dir(results_path + class_name)
        for file_name in os.listdir(class_path):
            try:
                file_path = class_path + "/" + file_name
                my_image = Image.open(file_path)
                black = Image.new('RGBA', my_image.size)
                my_image = Image.composite(my_image, black, my_image)
                cropped_image = my_image.crop(my_image.getbbox())
                cropped_image.save(f"{results_path}{class_name}/{file_name}")
                print(f"{file_name} done")
            except Exception as e:
                print(f"{file_name} failed {e}")
                continue



import base64
import asyncio
from fastapi import FastAPI, WebSocket

async def setup():
    print("Setup completed.")

async def run_command(command: str):
    print(f"Running command: {command}")
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=workdir
    )
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        line = line.decode().strip()
        print(line)
    await process.wait()
    

async def downloadModels():
    print(os.getcwd())
    await run_command("git clone https://github.com/IDEA-Research/GroundingDINO.git")
    await run_command("wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
    await run_command("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")


async def manually_check(ws: WebSocket):
    image_dir = workdir + 'DS_res'
    batch_size = 10
    print("Starting manual check...")
    images_files = []
    for filename in os.listdir(image_dir):
        images_files.extend([os.path.join(filename, f) for f in os.listdir(os.path.join(image_dir, filename)) if f.endswith(('.png', '.jpg'))])
    index = 0
    print(f"Total images to review: {len(images_files)}")
    while index < len(images_files):
        batch_files = images_files[index:index+batch_size]
        batch_data = []

        for filename in batch_files:
            path = os.path.join(image_dir, filename)
            with open(path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
            batch_data.append({"filename": filename, "data": encoded})
        
        # Send batch to app
        await ws.send_json({"type": "batch", "images": batch_data})
        print(f"Sent batch {index // batch_size + 1} with {len(batch_files)} images for review.")
        # Wait for response
        response = await ws.receive_json()
        print(f"received {response}")
        if response.get("action") == "finish":
            break
        
        index += batch_size

    print("Manual check completed.")


FUNCTION_REGISTRY = {
    "setup": setup,
    "crop": crop,
    "resize": resize,
    "segment": segment,
    "crop_precessed": crop_precessed,
    "manually_check": manually_check,
    "downloadModels": downloadModels,
}

app = FastAPI()
active_tasks = {}

async def execute_function(name: str):
    func = FUNCTION_REGISTRY.get(name)

    if not func:
        print(f"'{name}' function not found.")
        return

    print(f"▶️ Executing: {name}")

    try:
        result = func()
        if asyncio.iscoroutine(result):
            await result
        active_tasks.pop(name, None)
        print(f"✅ Finished: {name}")

    except Exception as e:
        print(f"🔥 Error in {name}: {str(e)}")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            data = json.loads(await ws.receive_text())

            if data["action"] == "run":
                tags = data.get("tags")
                for tag in tags:
                    tag_name = tag.get("name") if isinstance(tag, dict) else tag
                    if not tag_name:
                        continue

                    if tag_name == "manually_check":
                        await manually_check(ws)  
                        print(f"✅ Finished: {tag_name}")  
                        continue
                    else:
                        task = asyncio.create_task(execute_function(tag_name))
                        
                    active_tasks[tag_name] = task
            elif data["action"] == "delete":
                print("Received delete command")
                images_to_delete = data.get("images", [])
                image_dir = workdir + 'DS_res'
                for filename in images_to_delete:
                    path = os.path.join(image_dir, filename)
                    if os.path.exists(path):
                        os.remove(path)
                        print(f"Deleted: {filename}")

    except Exception:
        pass

if __name__ == '__main__':
    pass
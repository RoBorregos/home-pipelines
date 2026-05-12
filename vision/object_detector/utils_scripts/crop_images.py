from PIL import Image
import os
import json
import argparse
from processor import ProcessorSettings, Processor
from pydantic import BaseModel, Field
import torch
from asyncio import run, gather, Lock, Semaphore
import threading


class CropSettings(BaseModel):
    input_dir: str = Field(...,
                           description="Directory containing images to crop")
    output_dir: str = Field(...,
                            description="Directory to save cropped images")
    annotations_file: str = Field(...,
                                  description="JSON file with annotations for cropping")
    journal_dir: str = Field(...,
                             description="Directory to save the cropping journal")
    multi_threaded: bool = Field(
        False, description="Enable multi-threaded processing")


class CropProcessor:
    settings: CropSettings

    def __init__(self, settings: CropSettings):
        self.settings = settings
        self.processor_settings = ProcessorSettings(
            model_path="/home/ivan/home-pipelines/vision/object_detector/sam_vit_h_4b8939.pth",
            device="cuda" if torch.cuda.is_available() else "cpu",
            groundingdino_config_path="/home/ivan/home-pipelines/vision/object_detector/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            groundingdino_path="/home/ivan/home-pipelines/vision/object_detector/groundingdino_swint_ogc.pth",
            # sam2_path="/home/ivan/home-pipelines/vision/object_detector/sam2.1_hiera_large.pt",
            sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
            model_name="vit_h"
        )
        self.processor = Processor(ProcessorSettings=self.processor_settings)
        self.annotations = {}
        self.annotations_lock = Lock()

    def verify_directory_input(self):
        if not os.path.exists(self.settings.input_dir):
            raise FileNotFoundError(
                f"Input directory {self.settings.input_dir} does not exist.")
        # for i in os.scandir(self.settings.input_dir):
        #     if i.is_dir():
        #         print("Directory with subdirectories!")
        return True

    def prep_paths(self):
        if not os.path.exists(self.settings.output_dir):
            os.makedirs(self.settings.output_dir)
        if not os.path.exists(self.settings.journal_dir):
            os.makedirs(self.settings.journal_dir)
        if not os.path.exists(self.settings.annotations_file):
            with open(self.settings.annotations_file, 'w') as f:
                json.dump({}, f)

    async def process_dir(self, dir_path):
        classname = os.path.basename(dir_path)
        if not os.path.exists(os.path.join(self.settings.output_dir, classname)):
            os.makedirs(os.path.join(self.settings.output_dir, classname))
        if not os.path.exists(os.path.join(self.settings.journal_dir, classname)):
            os.makedirs(os.path.join(self.settings.journal_dir, classname))

        with open(os.path.join(self.settings.journal_dir, f"{classname}.txt"), 'w') as f:
            f.write(f"Processing directory: {dir_path}\n")
        with open(os.path.join(self.settings.journal_dir, f"{classname}.txt"), 'a') as f:
            f.write(
                f"Output directory: {os.path.join(self.settings.output_dir, classname)}\n")
            f.write(f"Processing {os.listdir(dir_path)} files\n")

        for filename in os.listdir(dir_path):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".heic"):
                img = Image.open(os.path.join(
                    dir_path, filename)).convert("RGB")
                res = self.processor.process_image(img, classname)
                org_filename = filename.split('.')[0]
                for indx, i in enumerate(res):
                    i.image.save(os.path.join(
                        self.settings.output_dir, classname, f"{org_filename}_{indx}.png"), "PNG")
                print(f"Processed {filename} in {classname}")
                with open(os.path.join(self.settings.journal_dir, f"{classname}.txt"), 'a') as f:
                    f.write(f"Processed {filename} in {classname}\n")
                async with self.annotations_lock:
                    self.annotations[os.path.join(classname, filename)] = [
                        {"bbox": i.bbox, "label": classname, "mask": str(i.mask.dumps()), "original": os.path.join(
                            classname, filename), "index": indx, "filename": f"{os.path.join(self.settings.output_dir, classname, f'{org_filename}_{indx}.png')}"}
                        for indx, i in enumerate(res)]

    def process(self):
        self.verify_directory_input()
        self.prep_paths()
        dirs = [d for d in os.scandir(self.settings.input_dir) if d.is_dir()]
        # tasks = [self.process_dir(d.path) for d in dirs]

        # async def _run_tasks():
        #     await gather(*tasks)
        # run(_run_tasks())
        # at top of file, add:
        # from asyncio import run, gather, Lock, Semaphore

        if self.settings.multi_threaded:
            # Use a thread pool for true parallelism
            import concurrent.futures

            max_workers = min(len(dirs), os.cpu_count() or 4)
            max_workers = 2
            max_workers = max(max_workers, 1)
            print(
                f"Processing {len(dirs)} directories with {max_workers} workers")
            print(
                f"Directories to process: {[os.path.basename(d.path) for d in dirs]}")

            # Create a thread pool executor
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Each directory gets processed in its own thread with its own event loop
                    futures = {executor.submit(lambda p: run(
                        self.process_dir(p)), d.path): d.path for d in dirs}

                    # Wait for all futures to complete
                    for future in concurrent.futures.as_completed(futures):
                        dir_path = futures[future]
                        dir_name = os.path.basename(dir_path)
                        try:
                            future.result()
                            print(f"Completed directory: {dir_name}")
                        except Exception as e:
                            print(f"Error processing {dir_name}: {str(e)}")
            except KeyboardInterrupt:
                print("\nCaught Ctrl+C! Shutting down gracefully...")
                return
        else:
            # fully sequential
            for d in dirs:
                run(self.process_dir(d.path))

        with open(self.settings.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)


def main():
    settings = CropSettings(
        input_dir="/home/ivan/home-pipelines/vision/object_detector/images",
        output_dir="/home/ivan/home-pipelines/vision/object_detector/cropped_images",
        annotations_file="/home/ivan/home-pipelines/vision/object_detector/annotations.json",
        journal_dir="/home/ivan/home-pipelines/vision/object_detector/crop_journal",
        multi_threaded=True
    )
    processor = CropProcessor(settings)
    processor.process()


if __name__ == "__main__":
    main()


# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# import sys
# sys.path.append("./")


# hols
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from pydantic import BaseModel, Field, ConfigDict
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from segment_anything import sam_model_registry, SamPredictor
import torch
from PIL import Image, ImageDraw
import cv2
import numpy as np


class ProcessorSettings(BaseModel):
    """
    Settings for the object detector processor.
    """
    model_name: str = Field(
        default="vit_hs",
        description=""
    )
    model_path: str = Field(
        default=""
    )

    sam2_path: str = Field(
        default="sam2.1_hiera_large.pt",
    )
    sam2_config: str = Field(
        default="sam2.1_hiera_l.yaml",
    )

    sam_2: bool = Field(
        default=True,
    )

    groundingdino_config_path: str = Field(
        default="GroundingDino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        description="Path to the GroundingDino configuration file."
    )
    groundingdino_path: str = Field(
        default="groundingdino_swint_ogc.pth"
    )

    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    token_spans: list | None = Field(
        default=None,
        description="Token spans for groundingdino. Should be a string representation of a list of tuples, e.g. '[(0, 4), (5, 9)]'."
    )
    box_threshold: float = Field(
        default=0.3,
        description="Threshold for filtering boxes based on their scores."
    )
    text_threshold: float = Field(
        default=0.25,
        description="Threshold for filtering text based on their scores."
    )
#     box_threshold = 0.3
# text_threshold = 0.25
# token_spans = None


class CroppedResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    image: Image.Image | None = None
    original_image: Image.Image | None = None
    bbox: torch.Tensor = torch.tensor([])
    mask: np.ndarray = np.array([])
    debug_image: Image.Image | None = None


class Processor:
    def __init__(self, ProcessorSettings: ProcessorSettings):
        self.settings = ProcessorSettings
        # self.sam_model = sam_model_registry[ProcessorSettings.model_name](
        #     checkpoint=ProcessorSettings.model_path)
        _reg = sam_model_registry[ProcessorSettings.model_name](
            checkpoint=ProcessorSettings.model_path)
        _reg.to(device=ProcessorSettings.device)
        if self.settings.sam_2:
            print("Using SAM2 model.")
            print(f"Using SAM2 config: {self.settings.sam2_config}")
            print(f"Using SAM2 path: {self.settings.sam2_path}")
            self.sam_predictor2 = SAM2ImagePredictor(
                build_sam2(
                    self.settings.sam2_config,
                    self.settings.sam2_path
                )
            )
        else:
            self.sam_predictor = SamPredictor(_reg)

        self.groundingdino_model = self.load_groundingdino_model(
            ProcessorSettings.groundingdino_config_path,
            ProcessorSettings.groundingdino_path
        )
        self.device = ProcessorSettings.device or "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def load_groundingdino_model(model_config_path, model_checkpoint_path) -> torch.nn.Module:
        args = SLConfig.fromfile(model_config_path)
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False)
        if False:
            print(f"Load model from {model_checkpoint_path}, "
                  f"load result: {load_res}")
        _ = model.eval()
        return model.to(args.device)

    def image_to_tensor_gdino(self, image: Image.Image) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        res, _ = transform(image, None)
        return res

    @staticmethod
    def format_caption(caption: str) -> str:
        res = caption.lower()
        res = res.strip()
        if not res.endswith("."):
            res = res + "."

        return res

    def predict_groundingdino(self, image: Image.Image, caption: str, box_threshold: float | None = None,
                              text_threshold: float | None = None, token_spans=None, with_logits: bool = True):
        assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"

        image_tensor = self.image_to_tensor_gdino(image)
        fixed_caption = self.format_caption(caption)

        image_ = image_tensor.to(self.settings.device)

        with torch.no_grad():
            outputs = self.groundingdino_model(
                image_[None],
                captions=[fixed_caption]
            )

        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        if token_spans is None:
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            tokenizer = self.groundingdino_model.tokenizer
            tokenized_caption = tokenizer(fixed_caption)

            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(
                    logit > text_threshold, tokenized_caption, tokenizer
                )

                if with_logits:
                    pred_phrases.append(
                        pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)

        else:
            positive_maps = create_positive_map_from_span(
                self.groundingdino_model.tokenizer(caption),
                token_span=token_spans
            ).to(self.settings.device)

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

    def oneonone_image_crop(self, image: Image.Image) -> Image.Image:
        """
        Crop the image to content.
        """
        min_x = 0
        min_y = 0
        max_x = image.width
        max_y = image.height
        # Convert image to numpy array
        img_array = np.array(image)

        for y in range(img_array.shape[0]):
            if np.any(img_array[y, :, :]):
                min_y = y
                break
            max_y = y
        for y in range(img_array.shape[0] - 1, -1, -1):
            if np.any(img_array[y, :, :]):
                max_y = y + 1
                break
        for x in range(img_array.shape[1]):
            if np.any(img_array[:, x, :]):
                min_x = x
                break
            max_x = x
        for x in range(img_array.shape[1] - 1, -1, -1):
            if np.any(img_array[:, x, :]):
                max_x = x + 1
                break
        # Crop the image using the calculated coordinates
        cropped_image = image.crop((min_x, min_y, max_x, max_y))
        return cropped_image

    def process_image(self, image: Image.Image, classname: str) -> list[CroppedResult]:
        # Set the text_threshold to None if token_spans is set.
        text_threshold = self.settings.text_threshold
        token_spans = self.settings.token_spans
        if token_spans is not None:
            text_threshold = None
            print("Using token_spans. Set the text_threshold to None.")

        # Handle token_spans evaluation safely
        token_spans_eval = None
        if token_spans is not None:
            try:
                token_spans_eval = eval(f"{token_spans}")
            except Exception:
                token_spans_eval = None

        boxes, labels = self.predict_groundingdino(
            image, classname, self.settings.box_threshold,
            text_threshold=text_threshold,
            token_spans=token_spans_eval)

        # Get image dimensions - note: PIL Image.size returns (width, height)
        size = image.size
        W, H = size[0], size[1]  # W=width, H=height

        assert len(boxes) == len(
            labels), "boxes and labels must have same length"

        # Convert PIL image to cv2 format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        cropped_results: list[CroppedResult] = []

        for box, label in zip(boxes, labels):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]

            # Apply padding
            padding = 10
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0)-padding, int(y0) - \
                padding, int(x1)+padding, int(y1)+padding

            # Validate if the bounding box is inside the image
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 > W:
                x1 = W
            if y1 > H:
                y1 = H

            sam_bounding_box = np.array([x0, y0, x1, y1])

            if self.settings.sam_2:
                with torch.inference_mode(), torch.autocast(device_type=self.settings.device, dtype=torch.bfloat16):
                    self.sam_predictor2.set_image(img)
                    mask, _, _ = self.sam_predictor2.predict(
                        point_coords=None,
                        point_labels=None,
                        box=sam_bounding_box,
                        multimask_output=False,
                    )
            else:
                self.sam_predictor.set_image(img)

                mask, _, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=sam_bounding_box,
                    multimask_output=False,
                )

            # Threshold input image using otsu thresholding as mask and refine with morphology
            ret, pngmask = cv2.threshold(mask[0].astype(
                np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            kernel = np.ones((9, 9), np.uint8)
            pngmask = cv2.morphologyEx(pngmask, cv2.MORPH_CLOSE, kernel)
            pngmask = cv2.morphologyEx(pngmask, cv2.MORPH_OPEN, kernel)

            # Create result image with alpha channel
            result = img.copy()
            result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = pngmask

            # Convert back to PIL Image format
            img_res = Image.fromarray(
                cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))

            # img_res = self.oneonone_image_crop(img_res)
            img_res = img_res.crop((x0, y0, x1, y1))
            cropped_results.append(
                CroppedResult(
                    image=img_res,
                    original_image=image,
                    bbox=box,
                    label=label,
                    mask=mask
                )
            )
        return cropped_results

    def process_image_with_debug(self, image: Image.Image, classname: str) -> list[CroppedResult]:
        cropped_results = self.process_image(image, classname)

        debug_images = []
        for result in cropped_results:
            debug_image = Image.fromarray(result.image)
            draw = ImageDraw.Draw(debug_image)
            x0, y0, x1, y1 = result.bbox.int().tolist()
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, y0), result.label, fill="white")
            mask = result.mask[0].astype(np.uint8)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.reshape(-1, 2)
                contour = [(x, y) for x, y in contour]
                draw.polygon(contour, outline="blue", fill=None)
            debug_image = debug_image.convert("RGB")
            debug_image = np.array(debug_image)
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
            debug_image = cv2.resize(debug_image, (640, 480))
            debug_image = cv2.putText(debug_image, result.label, (x0, y0 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (255, 255, 255), 2, cv2.LINE_AA)
            debug_image = cv2.rectangle(debug_image, (x0, y0), (x1, y1),
                                        (0, 0, 255), 2)
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
            debug_image = Image.fromarray(debug_image)
            # debug_images.append(debug_image)
            debug_images.append(
                CroppedResult(
                    label=result.label,
                    debug_image=debug_image,
                    original_image=image,
                    bbox=result.bbox,
                    mask=result.mask,
                    image=result.image
                )
            )

        return debug_images

import math
import random

import cv2
import numpy as np
import PIL
import spaces
import torch
from diffusers.models import ControlNetModel
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from PIL import Image

from instant_photo_maker.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
from instant_photo_maker.style_template import styles

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Vibrant Color"

# Load face encoder
app = FaceAnalysis(name="antelopev2", root="./instant_photo_maker/", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = "./instant_photo_maker/checkpoints/ip-adapter.bin"
controlnet_path = "./instant_photo_maker/checkpoints/ControlNetModel"

# Load pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

base_model_path = "wangqixun/YamerMIX_v8"

pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
)
pipe.cuda()
pipe.load_ip_adapter_instantid(face_adapter)
pipe.image_proj_model.to("cuda")
pipe.unet.to("cuda")


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def run_for_examples(face_file, prompt, style, negative_prompt):
    return generate_image(face_file, None, prompt, negative_prompt, style, True, 30, 0.8, 0.8, 5, 42)


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image



def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + " " + negative


def check_input_image(face_image):
    if face_image is None:
        print("Cannot find any input face image! Please upload the face image")
        raise ("Cannot find any input face image! Please upload the face image")


@spaces.GPU
def generate_image(
    face_image_path,
    pose_image_path,
    prompt,
    negative_prompt,
    style_name,
    enhance_face_region,
    num_steps,
    identitynet_strength_ratio,
    adapter_strength_ratio,
    guidance_scale,
    seed,
):
    if prompt is None:
        prompt = "A photo of a pirates"

    # apply the style template
    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    face_image = load_image(face_image_path)
    face_image = resize_img(face_image)
    face_image_cv2 = convert_from_image_to_cv2(face_image)
    height, width, _ = face_image_cv2.shape

    # Extract face features
    face_info = app.get(face_image_cv2)

    if len(face_info) == 0:
        print("Cannot find any face in the image! Please upload another person image")
        raise ("Cannot find any face in the image! Please upload another person image")

    face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1])[
        -1
    ]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])

    if pose_image_path is not None:
        pose_image = load_image(pose_image_path)
        pose_image = resize_img(pose_image)
        pose_image_cv2 = convert_from_image_to_cv2(pose_image)

        face_info = app.get(pose_image_cv2)

        if len(face_info) == 0:
            print("Cannot find any face in the reference image! Please upload another person image")
            raise ("Cannot find any face in the reference image! Please upload another person image")

        face_info = face_info[-1]
        face_kps = draw_kps(pose_image, face_info["kps"])

        width, height = face_kps.size

    if enhance_face_region:
        control_mask = np.zeros([height, width, 3])
        x1, y1, x2, y2 = face_info["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        control_mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(control_mask.astype(np.uint8))
    else:
        control_mask = None

    generator = torch.Generator(device=device).manual_seed(seed)

    print("Start inference...")
    print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

    pipe.set_ip_adapter_scale(adapter_strength_ratio)
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        control_mask=control_mask,
        controlnet_conditioning_scale=float(identitynet_strength_ratio),
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images

    return images[0]

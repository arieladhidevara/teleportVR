from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests, time, os, socket

def _detect_lan_ip(default="127.0.0.1"):
    # robust way to get your LAN IP (works without sending traffic)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return default

# Backend ↔ ComfyUI stays local on the PC:
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "http://127.0.0.1:8188")

# Client (PC/Quest) must fetch images via your LAN IP:
DEFAULT_PUBLIC = f"http://{_detect_lan_ip()}:8188"
PUBLIC_HOST    = os.getenv("PUBLIC_HOST", DEFAULT_PUBLIC)

MODEL_NAME     = os.getenv("MODEL_NAME", "sd_xl_base_1.0.safetensors")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class GenReq(BaseModel):
    prompt: str

def build_txt2img_pano_graph(prompt: str):
    return {
        "3": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": MODEL_NAME}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["3", 1]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "painting, illustration, cartoon, abstract, anime, low quality, blurry, text, watermark, deformed",
                         "clip": ["3", 1]}},
        "6": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 512, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["3", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "steps": 24, "cfg": 6.5, "denoise": 1.0, "seed": 0}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["3", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "pano"}}
    }

@app.post("/generate360")
def generate_360(req: GenReq):
    p = req.prompt.strip()
    if not p:
        return {"error": "empty_prompt"}

    graph = build_txt2img_pano_graph(p)
    payload = {"prompt": graph, "client_id": "unity_client"}

    r = requests.post(f"{COMFYUI_HOST}/prompt", json=payload, timeout=60)
    if r.status_code != 200:
        return {"error": "comfyui_prompt_failed", "status": r.status_code, "text": r.text}

    data = r.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        return {"error": "no_prompt_id_from_comfyui", "raw": data}

    # Poll history until the SaveImage node writes a file
    filename = subfolder = None
    for _ in range(240):
        h = requests.get(f"{COMFYUI_HOST}/history/{prompt_id}", timeout=30)
        if h.ok:
            hist = h.json()
            if prompt_id in hist:
                out = hist[prompt_id]["outputs"]
                if "9" in out and "images" in out["9"] and out["9"]["images"]:
                    filename = out["9"]["images"][0]["filename"]
                    subfolder = out["9"]["images"][0]["subfolder"]
                    break
        time.sleep(1)

    if not filename:
        return {"error": "generation_timeout_or_failed"}

    image_url = f"{PUBLIC_HOST}/view?filename={filename}&subfolder={subfolder}&type=output"
    return {"image_url": image_url}


@app.get("/health")
def health():
    return {"ok": True}
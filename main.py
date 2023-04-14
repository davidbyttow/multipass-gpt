from typing import Dict, List
from quart import Quart, request, jsonify, send_file

import quart_cors
import replicate

app = Quart(__name__)
quart_cors.cors(app, allow_origin="https://chat.openai.com")

# requires REPLICATE_API_TOKEN env var to be set

STABLE_DIFFUSION_MODEL = "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"
DAMO_TEXT_TO_VIDEO_MODEL = "cjwbw/damo-text-to-video:1e205ea73084bd17a0a3b43396e49ba0d6bc2e754e9283b2df49fad2dcf95755"
CONTROL_NET_SCRIBBLE_MODEL = "jagilley/controlnet-scribble:435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117"


@app.route("/create_image", methods=["POST"])
async def create_image():
    data = await request.json
    prompt = data["prompt"]

    # other options:
    # num_outputs=1
    # image_dimensions="768x768"
    # negative_prompt=""
    # num_inference_steps=50
    # guidance_scale=7.5
    # scheduler="DPMSolverMultistep"
    # seed=None

    output = replicate.run(
        STABLE_DIFFUSION_MODEL,
        input={"prompt": prompt},
    )

    if len(output) == 0:
        return jsonify({"status": "failed to create image"})

    image = output[0]
    return jsonify(
        {
            "status": "image created",
            "result": image,
        }
    )


@app.route("/create_image_from_drawing", methods=["POST"])
async def create_image_from_drawing():
    data = await request.json
    prompt = data["prompt"]
    image_url = data["image_url"]

    # other options:
    # num_samples=1
    # ddim_steps=20
    # scale=9
    # eta=None
    # a_prompt="best quality, extremely detailed"
    # n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    # seed=None

    output = replicate.run(
        CONTROL_NET_SCRIBBLE_MODEL,
        input={
            "image": image_url,
            "prompt": prompt,
        },
    )

    if len(output) != 2:
        return jsonify({"status": "failed to create image"})

    image = output[1]
    return jsonify(
        {
            "status": "Image created",
            "result": image,
        }
    )


@app.route("/create_video", methods=["POST"])
async def create_video():
    data = await request.json
    prompt = data["prompt"]

    # other options:
    # num_frames=16
    # num_inference_steps=50
    # fps=8
    # seed=None

    output = replicate.run(
        DAMO_TEXT_TO_VIDEO_MODEL,
        input={"prompt": prompt},
    )

    if not output:
        return jsonify({"status": "failed to create video"})

    return jsonify(
        {
            "status": "Video created",
            "url": output,
        }
    )


@app.route("/openapi.yaml", methods=["GET"])
async def get_openapi_yaml():
    return await send_file("openapi.yaml", mimetype="application/vnd.oai.openapi")


@app.route("/logo.png", methods=["GET"])
async def get_logo():
    return await send_file("logo.png", mimetype="image/png")


@app.route("/.well-known/ai-plugin.json", methods=["GET"])
async def get_ai_plugin_json():
    return await send_file("ai-plugin.json", mimetype="application/json")


@app.after_request
def add_header(r):
    # preventing cache bc chrome aggressively caches the file API calls
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

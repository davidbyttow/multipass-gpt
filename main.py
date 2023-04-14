from typing import Dict, List
from quart import Quart, request, jsonify, send_file

import replicate
import quart_cors

app = Quart(__name__)
quart_cors.cors(app, allow_origin="https://chat.openai.com")

# requires REPLICATE_API_TOKEN env var to be set

SD_MODEL_CHECKPOINT = "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"


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

    images = replicate.run(
        SD_MODEL_CHECKPOINT,
        input={"prompt": prompt},
    )

    if len(images) == 0:
        return jsonify({"status": "Failed to create image"})

    image = images[0]
    return jsonify(
        {
            "status": "Image created",
            "result": f"{image}",
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

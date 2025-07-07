import os
from flask import Flask, request, render_template, send_from_directory
from diffusers import StableDiffusionPipeline
import torch
from datetime import datetime

# Flask setup
app = Flask(__name__)
OUTPUT_FOLDER = "generated_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
print("Loading model... Please wait.")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded successfully.")

@app.route("/", methods=["GET", "POST"])
def index():
    generated_image_filename = None

    if request.method == "POST":
        prompt = request.form.get("prompt")

        if prompt:
            image = pipe(prompt).images[0]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"generated_{timestamp}.png"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            image.save(filepath)
            generated_image_filename = filename

    return render_template("index.html", image_filename=generated_image_filename)

@app.route("/generated_images/<filename>")
def get_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)

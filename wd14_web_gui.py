import csv
import os
import subprocess
import logging
import time
import numpy as np
import onnxruntime
from PIL import Image
from flask import Flask, render_template, request
import csv

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ONNX Model Path (Adjust as needed)
ONNX_MODEL_PATH = "./wd14_tagger_model/wd-v1-4-convnextv2-tagger-v2.onnx"

def load_onnx_model():
    """Loads the ONNX model."""
    try:
        sess = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
        return sess
    except Exception as e:
        log.error(f"Error loading ONNX model: {e}")
        return None

def preprocess_image(image_path):
    """Preprocesses the image for ONNX inference, maintaining aspect ratio."""
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        if width > height:
            new_width = 448
            new_height = int(height * (448 / width))
        else:
            new_height = 448
            new_width = int(width * (448 / height))

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        padded_image = Image.new("RGB", (448, 448))
        padded_image.paste(image, ((448 - new_width) // 2, (448 - new_height) // 2))

        image = np.array(padded_image).astype(np.float32) / 255.0
        image = np.transpose(image, (0, 1, 2))
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        log.error(f"Error preprocessing image: {e}")
        return None


def postprocess_output(output):
    """Extracts tags and scores from the ONNX model's output."""
    tags = []
    threshold = 0.35  # Adjust as needed

    # Load tag names from CSV file.
    tag_names = []
    try:
        with open("./wd14_tagger_model/selected_tags.csv", "r", newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header row

            for row in reader:
                tag_names.append(row[1]) #Extract the tag from the second column.

    except FileNotFoundError:
        log.error("CSV tag file not found.")
        return ""

    for i, score in enumerate(output[0]):
        if score > threshold:
            tags.append(tag_names[i])

    tags = [tag.strip() for tag in tags if tag.strip()] #Remove empty tags.

    return ", ".join(tags)


def run_onnx_inference(sess, image_path):
    """Runs ONNX inference and extracts tags."""
    try:
        image = preprocess_image(image_path)
        if image is None:
            return None

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        output = sess.run([output_name], {input_name: image})[0]

        print(f"Output Name: {output_name}")
        print(f"Output Shape: {output.shape}")

        tags = postprocess_output(output)
        return tags
    except Exception as e:
        log.error(f"Error running ONNX inference: {e}")
        return None

def caption_images(train_data_dir, caption_extension, general_threshold, character_threshold, repo_id, recursive, max_data_loader_n_workers, debug, undesired_tags, frequency_tags, always_first_tags, onnx, append_tags, force_download, caption_separator, tag_replacement, character_tag_expand, use_rating_tags, use_rating_tags_as_last_tag, remove_underscore, thresh):
    """Captions images using ONNX Runtime."""
    onnx_sess = load_onnx_model()
    if onnx_sess is None:
        return

    if not train_data_dir:
        log.error("Image folder is missing...")
        return

    if not caption_extension:
        log.error("Please provide an extension for the caption files.")
        return

    log.info(f"Captioning files in {train_data_dir}...")

    for root, _, files in os.walk(train_data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_path = os.path.join(root, file)
                tags = run_onnx_inference(onnx_sess, image_path)

                if tags and tags.strip(): #check if tags is not empty.
                    caption_file = os.path.splitext(image_path)[0] + caption_extension
                    if always_first_tags:
                        prefix_tags = [tag.strip() for tag in always_first_tags.split(",") if tag.strip()]
                        with open(caption_file, "w") as f:
                            f.write(", ".join(prefix_tags) + ", " + tags)
                    else:
                        with open(caption_file, "w") as f:
                            f.write(tags)
                    log.info(f"Captioned: {image_path}")
                elif always_first_tags: #If there are no tags, but there are always first tags, write those.
                    caption_file = os.path.splitext(image_path)[0] + caption_extension
                    prefix_tags = [tag.strip() for tag in always_first_tags.split(",") if tag.strip()]
                    with open(caption_file, "w") as f:
                        f.write(", ".join(prefix_tags))
                    log.info(f"Captioned: {image_path}")

        if not recursive:
            break

@app.route("/", methods=["GET", "POST"])
def index():
    """Handles the main web interface."""
    if request.method == "POST":
        caption_images(
            train_data_dir=request.form["train_data_dir"],
            caption_extension=request.form["caption_extension"],
            general_threshold=request.form["general_threshold"],
            character_threshold=request.form["character_threshold"],
            repo_id=request.form["repo_id"],
            recursive=request.form.get("recursive"),
            max_data_loader_n_workers=request.form["max_data_loader_n_workers"],
            debug=request.form.get("debug"),
            undesired_tags=request.form["undesired_tags"],
            frequency_tags=request.form.get("frequency_tags"),
            always_first_tags=request.form["always_first_tags"],
            onnx=request.form.get("onnx"),
            append_tags=request.form.get("append_tags"),
            force_download=request.form.get("force_download"),
            caption_separator=request.form["caption_separator"],
            tag_replacement=request.form["tag_replacement"],
            character_tag_expand=request.form.get("character_tag_expand"),
            use_rating_tags=request.form.get("use_rating_tags"),
            use_rating_tags_as_last_tag=request.form.get("use_rating_tags_as_last_tag"),
            remove_underscore=request.form.get("remove_underscore"),
            thresh=request.form["thresh"],
        )
        return "Captioning process started."
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

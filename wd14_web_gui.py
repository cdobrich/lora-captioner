import os
import logging
import numpy as np
import onnxruntime
from PIL import Image
from flask import Flask, render_template, request
import csv

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ONNX_MODEL_PATH = "./wd14_tagger_model/SmilingWolf_wd-v1-4-convnextv2-tagger-v2/model.onnx"
TAGS_CSV_PATH = "./wd14_tagger_model/SmilingWolf_wd-v1-4-convnextv2-tagger-v2/selected_tags.csv"

def load_onnx_model():
    """Loads the ONNX model."""
    if not os.path.exists(ONNX_MODEL_PATH) or not os.path.exists(TAGS_CSV_PATH):
        error_message = "Error: ONNX model or tag file not found. Please run ./setup.sh to download the necessary files."
        log.error(error_message)
        return None, error_message

    try:
        sess = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
        return sess, None
    except Exception as e:
        error_message = f"Error loading ONNX model: {e}"
        log.error(error_message)
        return None, error_message

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
    threshold = 0.35

    tag_names = []
    try:
        if not os.path.exists(TAGS_CSV_PATH):
            log.error(f"CSV tag file not found at: {TAGS_CSV_PATH}")
            return None

        with open(TAGS_CSV_PATH, "r", newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)

            for row in reader:
                tag_names.append(row[1])

    except FileNotFoundError:
        log.error("CSV tag file not found.")
        return None
    except Exception as e:
        log.error(f"Error reading CSV file: {e}")
        return None

    for i, score in enumerate(output[0]):
        if score > threshold:
            tags.append(tag_names[i])

    tags = [tag.strip() for tag in tags if tag.strip()]

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

        tags = postprocess_output(output)
        return tags
    except Exception as e:
        log.error(f"Error running ONNX inference: {e}")
        return None

def caption_images(train_data_dir, caption_extension, general_threshold, character_threshold, repo_id, recursive, max_data_loader_n_workers, debug, undesired_tags, frequency_tags, always_first_tags, onnx, append_tags, force_download, caption_separator, tag_replacement, character_tag_expand, use_rating_tags, use_rating_tags_as_last_tag, remove_underscore, thresh):
    """Captions images using ONNX Runtime."""
    onnx_sess, onnx_error = load_onnx_model()
    if onnx_sess is None:
        return onnx_error

    if not train_data_dir:
        return "Error: Image folder is missing."

    if not caption_extension:
        return "Error: Please provide an extension for the caption files."

    if not os.path.exists(train_data_dir):
        return f"Error: Directory '{train_data_dir}' does not exist."

    image_files = []
    for root, _, files in os.walk(train_data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_files.append(os.path.join(root, file))
        if not recursive:
            break

    if not image_files:
        return f"Error: No images found in '{train_data_dir}'."

    log.info(f"Captioning files in {train_data_dir}...")

    for image_path in image_files:
        tags = run_onnx_inference(onnx_sess, image_path)

        if tags:
            caption_file = os.path.splitext(image_path)[0] + caption_extension
            try:
                with open(caption_file, "w") as f:
                    if always_first_tags:
                        prefix_tags = [tag.strip() for tag in always_first_tags.split(",") if tag.strip()]
                        f.write(", ".join(prefix_tags) + (", " + tags if tags.strip() else ""))
                    else:
                        f.write(tags)
                log.info(f"Captioned: {image_path}")
            except PermissionError:
                return f"Error: Permission denied writing to '{caption_file}'."
        elif always_first_tags:
            caption_file = os.path.splitext(image_path)[0] + caption_extension
            try:
                prefix_tags = [tag.strip() for tag in always_first_tags.split(",") if tag.strip()]
                with open(caption_file, "w") as f:
                    f.write(", ".join(prefix_tags))
                log.info(f"Captioned: {image_path}")
            except PermissionError:
                return f"Error: Permission denied writing to '{caption_file}'."

    return "Captioning process completed."

@app.route("/", methods=["GET", "POST"])
def index():
    """Handles the main web interface."""
    error_message = None
    if request.method == "POST":
        repo_id = request.form.get("repo_id", "")
        train_data_dir = request.form.get("train_data_dir", "")
        caption_extension = request.form.get("caption_extension", "")
        general_threshold = request.form.get("general_threshold", "0.5")
        character_threshold = request.form.get("character_threshold", "0.5")
        max_data_loader_n_workers = request.form.get("max_data_loader_n_workers", "1")
        undesired_tags = request.form.get("undesired_tags", "")
        always_first_tags = request.form.get("always_first_tags", "")
        caption_separator = request.form.get("caption_separator", ", ")
        tag_replacement = request.form.get("tag_replacement", "")
        thresh = request.form.get("thresh", "0.5")

        error_message = caption_images(
            train_data_dir=train_data_dir,
            caption_extension=caption_extension,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            repo_id=repo_id,
            recursive=request.form.get("recursive"),
            max_data_loader_n_workers=max_data_loader_n_workers,
            debug=request.form.get("debug"),
            undesired_tags=undesired_tags,
            frequency_tags=request.form.get("frequency_tags"),
            always_first_tags=always_first_tags,
            onnx=request.form.get("onnx"),
            append_tags=request.form.get("append_tags"),
            force_download=request.form.get("force_download"),
            caption_separator=caption_separator,
            tag_replacement=tag_replacement,
            character_tag_expand=request.form.get("character_tag_expand"),
            use_rating_tags=request.form.get("use_rating_tags"),
            use_rating_tags_as_last_tag=request.form.get("use_rating_tags_as_last_tag"),
            remove_underscore=request.form.get("remove_underscore"),
            thresh=thresh,
        )
        if error_message is None:
            error_message = "Captioning process completed."

    return render_template("index.html", error_message=error_message, train_data_dir=request.form.get("train_data_dir", ""))

if __name__ == "__main__":
    app.run(debug=True)

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

MODEL_DIR = "./wd14_tagger_model/"
RECORD_FILE = os.path.join(MODEL_DIR, "model_choice.txt")

# List of character tags (modify as needed)
CHARACTER_TAGS = [
    "1girl", "1boy", "2girls", "2boys", #Add more character tags here.
]

def load_model_paths():
    """Loads the model and tags paths based on the user's choice."""
    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            repo_id = f.read().strip()
            log.info(f"Read repo_id from model_choice.txt: {repo_id}") #added log
            model_dir = os.path.join(MODEL_DIR, repo_id.replace("/", "_"))
            model_path = os.path.join(model_dir, "model.onnx")
            tags_path = os.path.join(model_dir, "selected_tags.csv")

            if not os.path.exists(model_path) or not os.path.exists(tags_path):
                raise FileNotFoundError(f"Model or tags file not found at: {model_path} or {tags_path}")

            return model_path, tags_path, None
    except FileNotFoundError as e:
        log.error(f"Model choice record file or model files not found: {e}. Using default model.")
        default_repo_id = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
        default_model_dir = os.path.join(MODEL_DIR, default_repo_id.replace("/", "_"))
        default_model_path = os.path.join(default_model_dir, "model.onnx")
        default_tags_path = os.path.join(default_model_dir, "selected_tags.csv")

        if not os.path.exists(default_model_path) or not os.path.exists(default_tags_path):
            return None, None, "Default model files not found."

        return default_model_path, default_tags_path, None

    except Exception as e:
        log.error(f"Error loading model paths: {e}")
        return None, None, str(e)
    
    

def load_onnx_model():
    """Loads the ONNX model for inference."""
    model_path, _, error = load_model_paths()
    if error:
        return None, error

    try:
        sess = onnxruntime.InferenceSession(model_path)
        return sess, None
    except Exception as e:
        log.error(f"Error loading ONNX model: {e}")
        return None, str(e)

def preprocess_image(image_path):
    """Preprocesses the image for ONNX inference, maintaining aspect ratio, and black padding."""
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

        padded_image = Image.new("RGB", (448, 448), (0, 0, 0)) # Black padding
        padded_image.paste(image, ((448 - new_width) // 2, (448 - new_height) // 2))

        image = np.array(padded_image).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0) # Add batch dimension (N, H, W, C)
        return image
    except Exception as e:
        log.error(f"Error preprocessing image: {e}")
        return None

def postprocess_output(output, general_threshold, character_threshold):
    """Extracts tags and scores from the ONNX model's output, using different thresholds."""
    tags = []
    tag_names = []
    _, tags_path, error = load_model_paths()
    if error:
        return None

    try:
        with open(tags_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            tag_names = [row[1] for row in reader]

    except FileNotFoundError:
        log.error("CSV tag file not found.")
        return None
    except Exception as e:
        log.error(f"Error reading CSV file: {e}")
        return None

    print(f"Output shape: {np.array(output).shape}")
    print(f"Output: {output}")

    for i, score_array in enumerate(output[0][0]):
        tag_name = tag_names[i]
        if tag_name in CHARACTER_TAGS:
            threshold = float(character_threshold)
        else:
            threshold = float(general_threshold)

        if score_array > threshold:
            tags.append(tag_name)

    tags = [tag.strip() for tag in tags if tag.strip()]
    return ", ".join(tags)

def run_onnx_inference(sess, image_path, general_threshold, character_threshold):
    """Runs ONNX inference and extracts tags."""
    try:
        image = preprocess_image(image_path)
        if image is None:
            return None

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        output = sess.run([output_name], {input_name: image})
        tags = postprocess_output(output, general_threshold, character_threshold)
        return tags
    except Exception as e:
        log.error(f"Error running ONNX inference: {e}")
        return None

def caption_images(train_data_dir, caption_extension, repo_id, recursive, max_data_loader_n_workers, debug, undesired_tags, frequency_tags, always_first_tags, onnx, append_tags, force_download, caption_separator, tag_replacement, character_tag_expand, use_rating_tags, use_rating_tags_as_last_tag, remove_underscore, general_threshold, character_threshold):
    """Captions images using ONNX Runtime."""
    onnx_sess, onnx_error = load_onnx_model()
    if onnx_sess is None:
        return onnx_error

    image_files = []
    if os.path.isdir(train_data_dir):
        for filename in os.listdir(train_data_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_files.append(os.path.join(train_data_dir, filename))
    else:
        return f"Error: '{train_data_dir}' is not a valid directory."

    for image_path in image_files:
        tags = run_onnx_inference(onnx_sess, image_path, general_threshold, character_threshold)

        if tags:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            caption_file = os.path.join(train_data_dir, base_name + caption_extension)

            with open(caption_file, "w", encoding="utf-8") as f:
                f.write(tags)

    return "Captioning process completed."


@app.route("/", methods=["GET", "POST"])
def index():
    """Handles the main web interface."""
    error_message = None
    selected_repo = None
    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            selected_repo = f.read().strip()
            log.info(f"Read repo_id from model_choice.txt for display: {selected_repo}") #added log
    except Exception as e:
        log.error(f"Error reading model_choice.txt: {e}")
        pass
    
    if request.method == "POST":
        train_data_dir = request.form.get("train_data_dir", "")
        caption_extension = request.form.get("caption_extension", ".txt")
        undesired_tags = request.form.get("undesired_tags", "")
        always_first_tags = request.form.get("always_first_tags", "")
        caption_separator = request.form.get("caption_separator", ", ")
        tag_replacement = request.form.get("tag_replacement", "")
        batch_size = request.form.get("batch_size", "1")
        general_threshold = request.form.get("general_threshold", "0.35")
        character_threshold = request.form.get("character_threshold", "0.7")
        max_data_loader_n_workers = request.form.get("max_data_loader_n_workers", "2")
        frequency_tags = request.form.get("frequency_tags") == "on"
        append_tags = request.form.get("append_tags") == "on"
        remove_underscore = request.form.get("remove_underscore") == "on"

        error_message = caption_images(
            train_data_dir=train_data_dir,
            caption_extension=caption_extension,
            repo_id=None,
            recursive=False,
            max_data_loader_n_workers=max_data_loader_n_workers,
            debug=False,
            undesired_tags=undesired_tags,
            frequency_tags=frequency_tags,
            always_first_tags=always_first_tags,
            onnx=True,
            append_tags=append_tags,
            force_download=False,
            caption_separator=caption_separator,
            tag_replacement=tag_replacement,
            character_tag_expand=False,
            use_rating_tags=False,
            use_rating_tags_as_last_tag=False,
            remove_underscore=remove_underscore,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
        )
        if error_message is None:
            error_message = "Captioning process completed."

    return render_template("index.html", error_message=error_message, train_data_dir=request.form.get("train_data_dir", ""), selected_repo=selected_repo)


if __name__ == "__main__":
    app.run(debug=True)

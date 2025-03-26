from flask import Flask, render_template, request, jsonify
import subprocess
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Helper functions (get_executable_path, setup_environment, add_pre_postfix) - same as before
def get_executable_path(name):
    paths = [
        os.path.join(os.getcwd(), name),
        os.path.join(os.getcwd(), "venv", "Scripts", name),
        os.path.join(os.getcwd(), "venv", "bin", name),
        os.path.join(os.getcwd(), "installer_files", "env", "Scripts", name),
        os.path.join(os.getcwd(), "installer_files", "env", "bin", name),
        os.path.join(os.getcwd(), "sd-scripts", name),
        os.path.join(os.getcwd(), "sd-scripts", "venv", "Scripts", name),
        os.path.join(os.getcwd(), "sd-scripts", "venv", "bin", name),
    ]

    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Executable {name} not found.")

def setup_environment():
    env = os.environ.copy()
    venv_dir = os.path.join(os.getcwd(), "venv")
    sd_scripts_venv_dir = os.path.join(os.getcwd(), "sd-scripts", "venv")
    installer_venv_dir = os.path.join(os.getcwd(), "installer_files", "env")

    if os.path.exists(venv_dir):
        env["PATH"] = os.path.join(venv_dir, "Scripts") + os.pathsep + env["PATH"]
    elif os.path.exists(sd_scripts_venv_dir):
        env["PATH"] = os.path.join(sd_scripts_venv_dir, "Scripts") + os.pathsep + env["PATH"]
    elif os.path.exists(installer_venv_dir):
      env["PATH"] = os.path.join(installer_venv_dir, "Scripts") + os.pathsep + env["PATH"]

    return env

def add_pre_postfix(folder, caption_file_ext, prefix, recursive):
    if not os.path.exists(folder):
        log.error(f"Folder {folder} does not exist.")
        return

    def process_file(file_path):
        if file_path.lower().endswith(caption_file_ext):
            try:
                with open(file_path, "r") as f:
                    content = f.read().strip()
                if prefix and not content.startswith(prefix.strip()):
                    if content:
                        new_content = f"{prefix.strip()}, {content}"
                    else:
                        new_content = prefix.strip()
                    with open(file_path, "w") as f:
                        f.write(new_content)
                log.info(f"Processed {file_path}")
            except Exception as e:
                log.error(f"Error processing {file_path}: {e}")

    if recursive:
        for root, _, files in os.walk(folder):
            for file in files:
                process_file(os.path.join(root, file))
    else:
        for file in os.listdir(folder):
            process_file(os.path.join(folder, file))

def caption_images(
    train_data_dir,
    caption_extension,
    batch_size,
    general_threshold,
    character_threshold,
    repo_id,
    recursive,
    max_data_loader_n_workers,
    debug,
    undesired_tags,
    frequency_tags,
    always_first_tags,
    onnx,
    append_tags,
    force_download,
    caption_separator,
    tag_replacement,
    character_tag_expand,
    use_rating_tags,
    use_rating_tags_as_last_tag,
    remove_underscore,
    thresh,
):
    if not train_data_dir:
        log.error("Image folder is missing...")
        return

    if not caption_extension:
        log.error("Please provide an extension for the caption files.")
        return

    repo_id_converted = repo_id.replace("/", "_")
    if not os.path.exists(f"./wd14_tagger_model/{repo_id_converted}"):
        force_download = True

    log.info(f"Captioning files in {train_data_dir}...")
    run_cmd = [
        rf'{get_executable_path("accelerate")}',
        "launch",
        rf"{os.path.join(os.getcwd(), 'sd-scripts/finetune/tag_images_by_wd14_tagger.py')}",
    ]

    if append_tags:
        run_cmd.append("--append_tags")
    run_cmd.append("--batch_size")
    run_cmd.append(str(int(batch_size)))
    run_cmd.append("--caption_extension")
    run_cmd.append(caption_extension)
    run_cmd.append("--caption_separator")
    run_cmd.append(caption_separator)

    if character_tag_expand:
        run_cmd.append("--character_tag_expand")
    if character_threshold != 0.35:
        run_cmd.append("--character_threshold")
        run_cmd.append(str(character_threshold))
    if debug:
        run_cmd.append("--debug")
    if force_download:
        run_cmd.append("--force_download")
    if frequency_tags:
        run_cmd.append("--frequency_tags")
    if general_threshold != 0.35:
        run_cmd.append("--general_threshold")
        run_cmd.append(str(general_threshold))
    run_cmd.append("--max_data_loader_n_workers")
    run_cmd.append(str(int(max_data_loader_n_workers)))

    if onnx:
        run_cmd.append("--onnx")
    if recursive:
        run_cmd.append("--recursive")
    if remove_underscore:
        run_cmd.append("--remove_underscore")
    run_cmd.append("--repo_id")
    run_cmd.append(repo_id)
    if tag_replacement:
        run_cmd.append("--tag_replacement")
        run_cmd.append(tag_replacement)
    if thresh != 0.35:
        run_cmd.append("--thresh")
        run_cmd.append(str(thresh))
    if undesired_tags:
        run_cmd.append("--undesired_tags")
        run_cmd.append(undesired_tags)
    if use_rating_tags:
        run_cmd.append("--use_rating_tags")
    if use_rating_tags_as_last_tag:
        run_cmd.append("--use_rating_tags_as_last_tag")

    run_cmd.append(rf"{train_data_dir}")

    env = setup_environment()

    command_to_run = " ".join(run_cmd)
    log.info(f"Executing command: {command_to_run}")

    subprocess.run(run_cmd, env=env)

    add_pre_postfix(
        folder=train_data_dir,
        caption_file_ext=caption_extension,
        prefix=always_first_tags,
        recursive=recursive,
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        train_data_dir = request.form['train_data_dir']
        caption_extension = request.form['caption_extension']
        batch_size = int(request.form['batch_size'])
        general_threshold = float(request.form['general_threshold'])
        character_threshold = float(request.form['character_threshold'])
        repo_id = request.form['repo_id']
        recursive = 'recursive' in request.form
        max_data_loader_n_workers = int(request.form['max_data_loader_n_workers'])
        debug = 'debug' in request.form
        undesired_tags = request.form['undesired_tags']
        frequency_tags = 'frequency_tags' in request.form
        always_first_tags = request.form['always_first_tags']
        onnx = 'onnx' in request.form
        append_tags = 'append_tags' in request.form
        force_download = 'force_download' in request.form
        caption_separator = request.form['caption_separator']
        tag_replacement = request.form['tag_replacement']
        character_tag_expand = 'character_tag_expand' in request.form
        use_rating_tags = 'use_rating_tags' in request.form
        use_rating_tags_as_last_tag = 'use_rating_tags_as_last_tag' in request.form
        remove_underscore = 'remove_underscore' in request.form
        thresh = float(request.form['thresh'])

        caption_images(
            train_data_dir,
            caption_extension,
            batch_size,
            general_threshold,
            character_threshold,
            repo_id,
            recursive,
            max_data_loader_n_workers,
            debug,
            undesired_tags,
            frequency_tags,
            always_first_tags,
            onnx,
            append_tags,
            force_download,
            caption_separator,
            tag_replacement,
            character_tag_expand,
            use_rating_tags,
            use_rating_tags_as_last_tag,
            remove_underscore,
            thresh,
        )

        return jsonify({'message': 'Captioning completed'})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

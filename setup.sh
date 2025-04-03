#!/bin/bash

# Function to download model and tags
download_model() {
  local REPO_ID="$1"
  local MODEL_DIR="./wd14_tagger_model/$(echo "$REPO_ID" | sed 's/\//_/g')"
  local MODEL_FILE="model.onnx"
  local TAGS_FILE="selected_tags.csv"
  local RECORD_FILE="./wd14_tagger_model/model_choice.txt"

  mkdir -p "$MODEL_DIR"

  echo
  echo "Gathering asset files for $REPO_ID"
  echo

  # Check model file
  if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Local ONNX model file found."
  else
    echo "Downloading ONNX model..."
    curl -L "https://huggingface.co/$REPO_ID/resolve/main/$MODEL_FILE" -o "$MODEL_DIR/$MODEL_FILE"
    echo "$REPO_ID" > "$MODEL_DIR/model.onnx.info"
  fi

  # Check tags file
  if [ -f "$MODEL_DIR/$TAGS_FILE" ]; then
    echo "Local tag list file found."
  else
    echo "Downloading tag list..."
    curl -L "https://huggingface.co/$REPO_ID/resolve/main/$TAGS_FILE" -o "$MODEL_DIR/$TAGS_FILE"
    echo "$REPO_ID" > "$MODEL_DIR/selected_tags.csv.info"
  fi

  echo "$REPO_ID" > "$RECORD_FILE" #record the users choice in the base directory.
  echo
}

# Present download options to the user
echo "Choose a model to download:"
echo "1. wd-v1-4-convnextv2-tagger-v2 (default)"
echo "2. wd-vit-large-tagger-v3"
read -p "Enter your choice (1 or 2): " choice

case "$choice" in
  1|"")
    download_model "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
    ;;
  2)
    download_model "SmilingWolf/wd-vit-large-tagger-v3"
    ;;
  *)
    echo "Invalid choice. Downloading default model."
    download_model "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
    ;;
esac

# Create and activate a Python virtual environment
echo "Creating and activating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "Setup complete!"
echo
echo "You may run: bash start.sh"
echo

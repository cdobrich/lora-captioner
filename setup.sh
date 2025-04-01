#!/bin/bash

# Define the Hugging Face repository and file names
REPO_ID="SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
MODEL_FILE="model.onnx"
TAGS_FILE="selected_tags.csv"

# Define the local directory where the files should be stored
MODEL_DIR="./wd14_tagger_model/SmilingWolf_wd-v1-4-convnextv2-tagger-v2"

# Create the directories if they don't exist
mkdir -p "$MODEL_DIR"

echo
echo "Gathering asset files"
echo
# Check if the model file already exists
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
  echo "ONNX model already exists. Skipping download."
else
  # Download the model file
  echo "Downloading ONNX model..."
  curl -L "https://huggingface.co/$REPO_ID/resolve/main/$MODEL_FILE" -o "$MODEL_DIR/$MODEL_FILE"
fi

# Check if the tags file already exists
if [ -f "$MODEL_DIR/$TAGS_FILE" ]; then
  echo "Tag list already exists. Skipping download."
else
  # Download the tags file
  echo "Downloading tag list..."
  curl -L "https://huggingface.co/$REPO_ID/resolve/main/$TAGS_FILE" -o "$MODEL_DIR/$TAGS_FILE"
fi

echo

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
echo "You may run: python start.py"
echo

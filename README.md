# LoRA-Captioner for Non-Nvidia GPUs.

This is an image tagging tool for Stable Diffusion LoRAs for use on non-Nvidia GPUs. In theory it should even work on just a pure CPU. It was inspired by the [kohya_ss](https://github.com/bmaltais/kohya_ss) project. It uses the same resources, so this is like using the WD14 captioning tool.

(I have tested this on my Thinkpad T480 with Intel HD Graphics 620.)

I think I implemented all the currently available GUI features but I might have missed some.

# Installation Setup

This is designed to run on Linux, but in theory nothing is stopping you from running on Windows or MacOS. All you need is Python 3 installed, and CURL.

This downloads the ONNX model, the selected tags file (the same resources as Kohya_ss), and installs the captioner program python dependencies.

In a command console, run:

```
bash setup.sh
```

Once the setup is complete, move to the Run section.

# Usage / Running

To start the backend of the program, in a command console, run:

```
bash start.sh
```

This will launch the backend of the program in a console, running the program for your browser to load locally. 

Open this address URL in your browser of choice:

http://127.0.0.1:5000/

# Stopping the Program

To stop the program, press CTRL+C in the console window or just close the terminal console. You can close the browser tab also.

# Extra Info - Image Preprocessing Info and Explanation

Extra details being recorded in case they are ever needed in the future.

## Aspect Ratio Calculation:
- We get the original width and height of the image.
- We determine which dimension is larger and scale the other dimension proportionally to fit within 448 pixels.

# Resizing

- We resize the image using Image.Resampling.LANCZOS for better quality.

# Padding (Optional)

- We create a new black image of size 448x448.
- We paste the resized image into the center of the black image.
- This ensures that all input images are exactly 448x448, which is required by the ONNX model.

# Conversion to NumPy Array

- We convert the image to a NumPy array, normalize the pixel values, and adjust the color channel order.

# Important Considerations

- Padding Color: You can change the padding color by modifying the Image.new("RGB", (448, 448)) line.
- Performance: Padding adds an extra step, which may slightly impact performance.
- Alternative Padding: Alternatives to black padding include mirroring the image edges or using a blurred version of the image for padding.
- Model Performance: Padding can sometimes negatively affect model performance, depending on the model. Test with and without padding to see which produces better results for your use case.
- Cropping instead of padding: Cropping the image to the correct aspect ratio may be a better option than padding, depending on the image content.

# Development Unit Testing

For testing, you need to install `pytest` and `pytest-cov` and `pytest-timeout`.

## Run all tests
```
pytest
```

## Run specific test file

```
pytest tests/test_wd14_web_gui.py
```

## Run with coverage report

```
pytest --cov=wd14_web_gui tests/
```

## Test Information

Mocking:
- The `onnxruntime.InferenceSession`, `preprocess_image`, and `postprocess_output` functions are mocked to isolate the functions being tested and prevent actual ONNX inference.

Dummy Files:
- Dummy ONNX model and CSV files are created for testing purposes and cleaned up after each test.

Test Coverage:
- Tests are included for `load_onnx_model`, `preprocess_image`, `postprocess_output`, `run_onnx_inference`, and `caption_images`.
The index route is also tested.

Error Handling:
- Tests are included to ensure that error conditions are handled correctly.

Test for always_first_tags:
- Added a test to check if only the always_first_tags are correctly written to the caption file, when the onnx model returns no tags.

Pathing:
- The test now creates and removes a test directory, and test image, to prevent errors.

Flask Testing:
- Uses the Flask test client to test the web routes.


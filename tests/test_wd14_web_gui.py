import unittest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from wd14_web_gui import (
    load_onnx_model,
    preprocess_image,
    postprocess_output,
    run_onnx_inference,
    caption_images,
)


class TestWd14WebGui(unittest.TestCase):

    @patch("wd14_web_gui.os.path.exists", return_value=True)
    @patch("wd14_web_gui.onnxruntime.InferenceSession")
    def test_load_onnx_model_success(self, mock_inference_session, mock_exists):
        """Test successful loading of ONNX model."""
        mock_session = MagicMock()
        mock_inference_session.return_value = mock_session
        session, error = load_onnx_model()
        self.assertIsNotNone(session)
        self.assertIsNone(error)

    @patch("wd14_web_gui.os.path.exists", return_value=False)
    def test_load_onnx_model_missing_files(self, mock_exists):
        """Test loading failure due to missing model files."""
        session, error = load_onnx_model()
        self.assertIsNone(session)
        self.assertIn("Error", error)

    def test_preprocess_image_valid(self):
        """Test image preprocessing with a valid image."""
        test_image_path = "test_image.jpg"
        image = np.random.rand(448, 448, 3) * 255
        image = image.astype(np.uint8)

        from PIL import Image
        Image.fromarray(image).save(test_image_path)

        result = preprocess_image(test_image_path)
        os.remove(test_image_path)  # Cleanup test file

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, 448, 448, 3))

    def test_preprocess_image_invalid(self):
        """Test preprocessing with an invalid image file."""
        result = preprocess_image("non_existent.jpg")
        self.assertIsNone(result)

    def test_postprocess_output_valid(self):
        """Test postprocessing with valid output data."""
        output = np.array([[0.4, 0.2, 0.5, 0.1]])
        with patch("wd14_web_gui.open", unittest.mock.mock_open(read_data="index,tag\n0,cat\n1,dog\n2,flower\n3,sky")):
            tags = postprocess_output(output)
        self.assertIn("cat", tags)
        self.assertIn("flower", tags)

    def test_postprocess_output_missing_csv(self):
        """Test postprocessing when CSV is missing."""
        with patch("wd14_web_gui.os.path.exists", return_value=False):
            tags = postprocess_output([[0.4, 0.2, 0.5, 0.1]])
        self.assertIsNone(tags)

    @patch("wd14_web_gui.load_onnx_model", return_value=(MagicMock(), None))
    @patch("wd14_web_gui.run_onnx_inference", return_value="cat, dog")
    @patch("wd14_web_gui.os.path.exists", return_value=True)
    def test_caption_images_success(self, mock_exists, mock_inference, mock_model):
        """Test successful image captioning."""
        test_dir = "test_images"
        os.makedirs(test_dir, exist_ok=True)
        test_image_path = os.path.join(test_dir, "test.jpg")
        
        from PIL import Image
        Image.new("RGB", (100, 100)).save(test_image_path)

        result = caption_images(
            train_data_dir=test_dir,
            caption_extension=".txt",
            general_threshold=0.5,
            character_threshold=0.5,
            repo_id="",
            recursive=False,
            max_data_loader_n_workers=1,
            debug=False,
            undesired_tags="",
            frequency_tags="",
            always_first_tags="",
            onnx=True,
            append_tags=False,
            force_download=False,
            caption_separator=", ",
            tag_replacement="",
            character_tag_expand=False,
            use_rating_tags=False,
            use_rating_tags_as_last_tag=False,
            remove_underscore=False,
            thresh=0.35,
        )
        
        self.assertIn("Captioning process completed", result)
        
        import shutil  # Add this import

        # Clean up test files
        os.remove(test_image_path)
        shutil.rmtree(test_dir)  # This removes the directory and all its contents


    @patch("wd14_web_gui.load_onnx_model", return_value=(None, "Error loading model"))
    def test_caption_images_model_failure(self, mock_model):
        """Test captioning failure due to model load error."""
        result = caption_images(
            train_data_dir="test_images",
            caption_extension=".txt",
            general_threshold=0.5,
            character_threshold=0.5,
            repo_id="",
            recursive=False,
            max_data_loader_n_workers=1,
            debug=False,
            undesired_tags="",
            frequency_tags="",
            always_first_tags="",
            onnx=True,
            append_tags=False,
            force_download=False,
            caption_separator=", ",
            tag_replacement="",
            character_tag_expand=False,
            use_rating_tags=False,
            use_rating_tags_as_last_tag=False,
            remove_underscore=False,
            thresh=0.35,
        )
        self.assertIn("Error loading model", result)


if __name__ == "__main__":
    unittest.main()

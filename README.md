# Piano Note Detection Model Training Script
A deep learning model for real-time piano key detection from video frames. This model analyzes sequences of video frames to identify which of the 88 piano keys are currently being pressed.

## Overview

This repository contains a TensorFlow-based training pipeline that:
- Processes sequences of video frames to detect pressed piano keys
- Outputs predictions for all 88 piano keys (MIDI notes 21-108)
- Converts trained models to TFLite format for efficient deployment
- Integrates with Viam's ML training infrastructure for automated deployment

## Model Architecture

The model uses a hybrid 3D-2D convolutional neural network architecture:

- **Input**: 5 consecutive grayscale frames (480x640 pixels)
- **3D Convolution Stage**: Aggregates temporal information across frames
- **2D Processing**: Processes spatial features with residual blocks
- **Output**: 88 binary predictions (one per piano key)

### Key Features:
- Residual blocks with progressive downsampling
- Weighted binary cross-entropy loss to handle class imbalance
- Data augmentation (brightness, contrast, noise)
- Early stopping and learning rate reduction callbacks

## Requirements

- Python 3.7+
- TensorFlow >= 2.10.0
- Keras < 2.14.0
- OpenCV
- NumPy
- tqdm

Install dependencies:
```bash
pip install -e .
```

## Dataset Format

The training script expects a JSONLines file where each line contains:

```json
{
  "image_path": "path/to/image.jpg",
  "classification_annotations": [
    {"annotation_label": "21"},
    {"annotation_label": "45"},
    ...
  ],
  "bounding_box_annotations": [
    ... (not used)
  ]
}
```

- `image_path`: Path to the video frame
- `annotation_label`: MIDI note number (21-108) for pressed keys

## Training

### Local Training

Run the training script with:

```bash
python model/training.py \
  --dataset_file path/to/dataset.jsonl \
  --model_output_directory output/dir \
  --num_epochs 200
```

Parameters:
- `--dataset_file`: Path to JSONLines dataset file
- `--model_output_directory`: Directory for saving trained model
- `--num_epochs`: Number of training epochs (default: 200)

### Output Files

The training script generates:
- `piano_detection_model.tflite`: Quantized TFLite model
- `labels.txt`: List of MIDI note numbers (21-108)

## Model Performance

The model tracks several metrics during training:
- **Binary Accuracy**: Overall accuracy across all keys
- **Precision**: Ratio of correctly predicted pressed keys
- **Recall**: Ratio of detected pressed keys

The weighted loss function prioritizes detecting pressed keys (positive samples) over non-pressed keys to handle the natural class imbalance in piano playing.

## Deployment

### GitHub Actions Workflow

The repository includes automated deployment via GitHub Actions:

```yaml
name: Deploy ML Training Image
on:
  push:
    branches: [main]
```

This workflow:
1. Builds a Docker image with the training script
2. Deploys to Viam's ML training infrastructure
3. Registers the model as `note-classifier` with multi-label classification type

### Environment Variables

Required secrets for deployment:
- `VIAM_DEV_API_KEY_ID`: Viam API key ID
- `VIAM_DEV_API_KEY`: Viam API key
- `VIAM_DEV_ORG_ID`: Viam organization ID

## Technical Details

### Input Processing
- Loads 5 consecutive frames for temporal context
- Converts images to grayscale
- Resizes to 480x640 pixels
- Normalizes pixel values to [0, 1]

### Data Augmentation
- Random brightness adjustment (±20%)
- Random contrast adjustment (0.7-1.3x)
- Gaussian noise addition (σ=0.05)

### Model Optimization
- Uses Adam optimizer with initial learning rate of 0.001
- Implements early stopping (patience: 20 epochs)
- Reduces learning rate on plateau (factor: 0.5, patience: 10)
- Converts to TFLite with DEFAULT optimization

## Usage
In order to submit this script with custom arguments, you must use the Viam CLI. One such example is included below:
```
viam train submit custom with-upload --dataset-id=<DATASET-ID> --model-org-id=<ORG-ID> --model-name=classification --model-type=<DESIRED_TYPE> --framework=tflite --path=<REPO-TAR-PATH> --script-name=note_classifier --args=num_epochs=3 [...named_args]
```
Be sure to note that labels is surrounded with single quotes then enclosed with double quotes to ensure it is submitted properly.  
To use the training script from the Viam registry, use the below command.
```
viam train submit custom from-registry --dataset-id=<DATASET-ID> --org-id=<ORG-ID> --model-name=<model-name> --script-name=<script-name> --version=<version> --args=num_epochs=200 [...named_args]
```

## Workflows

### Pull Request

When you submit a pull request a workflow will run using our [common workflows](https://github.com/viam-modules/common-workflows/) that will lint check your code, build it in the docker image we us in production (tensorflow/tensorflow:2.11.1-gpu) and run the test file you specify.

The default test files is `scripts/test.sh`. If this changes you will need to update `.github/workflows/pull_request.yaml` so that it's

```
jobs:
  build:
    uses: viam-modules/common-workflows/.github/workflows/lint_and_test.yaml
    with:
      test_script_name: NEW_TEST_FILE_NAME
```

### Main

Upon merging to `main` a workflow will automatically update the module in `viam-dev` allowing for people to use your latest changes. The configs you can (but shouldn't!) play with are:
1. framework -- DO NOT CHANGE THIS! This is a tflite script and will always be (see: repo name)
2. script_name -- This is what the name will be in the registry. If you change this, it will make a new training script in the registry. Be aware
3. model_type -- single_label_classification

# PathoVision AI

AI-assisted digital pathology web application for the PCam/ResNet50 mini-project.

## Features
- Pathologist registration and login
- Case/patient reference entry
- Histopathology image upload
- ResNet50 prediction
- Confidence score
- Grad-CAM explainability heatmap
- Analysis history
- PDF report generation
- Dashboard statistics
- Research/decision-support disclaimer

## Important
The trained model file is NOT included in this ZIP because it is a separate ~94 MB file.
Copy your downloaded `resnet50_pcam.keras` into this project root.

The model shown in the project screenshots was trained for 224x224 inputs and binary PCam-style classification. The default preprocessing below uses pixel values scaled to [0,1]. If your training notebook used a different preprocessing function, change `MODEL_PREPROCESSING` in `utils/prediction.py` to match it exactly.

## Run
1. Create a virtual environment.
2. Install requirements.
3. Put `resnet50_pcam.keras` beside `app.py`.
4. Run `python app.py`.
5. Open http://127.0.0.1:5000

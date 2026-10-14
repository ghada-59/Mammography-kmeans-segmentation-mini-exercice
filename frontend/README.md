# Mammography Image Lab — React interface

This optional React/Vite frontend calls the existing K-Means and validation functions in main.py through a local FastAPI adapter.

## Setup

From the repository root, install the optional API dependencies:

~~~bash
python -m pip install -r requirements-react-api.txt
~~~

Start the API in terminal 1:

~~~bash
python -m uvicorn segmentation_api:app --reload --host 127.0.0.1 --port 8000
~~~

In terminal 2:

~~~bash
cd frontend
npm install
npm run dev
~~~

Open the local URL printed by Vite, normally http://127.0.0.1:5173/.

## Workflow

1. Upload a PNG, JPG, BMP, PGM or TIFF image.
2. Confirm the local preview.
3. Run K-Means for k = 2, 3 and 4.
4. Compare the segmentation maps, inertia and silhouette score.
5. Export a text report of the current run.

For responsive interaction, images with a dimension above 256 pixels are resized before segmentation. Results and metrics therefore describe the resized image used in the UI.

## Scope

The algorithm clusters grayscale pixel intensities. It does not identify or validate anatomical structures and is not a diagnostic system.

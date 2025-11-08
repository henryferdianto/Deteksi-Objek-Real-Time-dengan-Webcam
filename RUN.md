Run instructions (alternate)

1. Install requirements:

```powershell
python -m pip install -r requirements.txt
```

2. Download YOLOv3-tiny files into `./yolo` (or run the notebook `main.ipynb` which contains a cell to download them):
- `yolov3-tiny.weights` (approx 34 MB)
- `yolov3-tiny.cfg`
- `coco.names`

3. Run the detector:

```powershell
python detect.py
```

Press `q` in the display window to stop.

Notes:
- If your camera index is not 0, run `python detect.py --camera 1` (or other index).
- On Windows, ensure the terminal has permission to access the camera and no other app is using it.

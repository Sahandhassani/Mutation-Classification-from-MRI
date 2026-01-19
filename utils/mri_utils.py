import os
import numpy as np
import pydicom

def load_dicom_series_safe(series_dir):
    valid_slices = []
    skipped = 0

    for f in os.listdir(series_dir):
        if not f.lower().endswith(".dcm"):
            continue

        path = os.path.join(series_dir, f)
        try:
            ds = pydicom.dcmread(path)
            if not hasattr(ds, "PixelData"):
                skipped += 1
                continue

            img = ds.pixel_array
            valid_slices.append((ds, img))

        except Exception:
            skipped += 1

    if len(valid_slices) == 0:
        return None

    try:
        valid_slices.sort(key=lambda x: float(x[0].ImagePositionPatient[2]))
    except:
        valid_slices.sort(key=lambda x: int(x[0].InstanceNumber))

    volume = np.stack([img for _, img in valid_slices])
    return volume

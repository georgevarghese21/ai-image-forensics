"""EXIF and file metadata extraction.

    python -m src.metadata --image photo.jpg

IMPORTANT (spec section 7): metadata is supporting context, never proof.

    missing EXIF  != AI-generated
    camera EXIF   != real photograph

EXIF is stripped by screenshots, social platforms and any re-save, and can be
forged in one line of code. This module reports what is present. It does not
contribute to the verdict.
"""
import argparse
import json
from pathlib import Path

from PIL import ExifTags, Image

# Tags worth surfacing. Full EXIF runs to hundreds of fields, most irrelevant.
INTERESTING = {
    "Make", "Model", "LensModel", "LensMake", "Software", "DateTime",
    "DateTimeOriginal", "ExposureTime", "FNumber", "ISOSpeedRatings",
    "FocalLength", "Flash", "Orientation", "ColorSpace", "Artist", "Copyright",
}


def extract(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    result = {
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "exif_present": False,
        "gps_present": False,
        "camera": None,
        "lens": None,
        "software": None,
        "timestamp": None,
        "exif_fields": {},
        "notes": [],
    }

    with Image.open(path) as im:
        result["format"] = im.format
        result["mode"] = im.mode
        result["width"], result["height"] = im.size

        exif = im.getexif()
        if not exif:
            result["notes"].append(
                "No EXIF present. This is NOT evidence of AI generation - "
                "screenshots, social media uploads and re-saves all strip it.")
            return result

        result["exif_present"] = True
        tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}

        for name in INTERESTING:
            if name in tags:
                result["exif_fields"][name] = str(tags[name])[:100]

        # GPS lives in its own IFD, not the top-level tag dict.
        try:
            gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
            if gps:
                result["gps_present"] = True
                result["notes"].append(
                    "GPS data present. Consider privacy before sharing.")
        except Exception:
            pass

        make = tags.get("Make", "")
        model = tags.get("Model", "")
        camera = f"{make} {model}".strip()
        result["camera"] = camera or None
        result["lens"] = tags.get("LensModel")
        result["software"] = tags.get("Software")
        result["timestamp"] = tags.get("DateTimeOriginal") or tags.get("DateTime")

        if result["camera"]:
            result["notes"].append(
                "Camera EXIF present. This is NOT evidence the image is real - "
                "EXIF fields can be written to any file.")
        if result["software"]:
            result["notes"].append(
                f"Processed by: {result['software']}. Editing software in EXIF "
                "indicates modification, not origin.")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    print(json.dumps(extract(args.image), indent=2))


if __name__ == "__main__":
    main()

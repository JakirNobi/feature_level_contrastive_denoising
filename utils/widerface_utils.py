"""
Wider Face dataset conversion utilities.
Converts raw Wider Face annotations to YOLO format.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_wider_annotation(anno_path: str) -> Dict[str, List[List[int]]]:
    """
    Parse a Wider Face annotation file.

    Args:
        anno_path: Path to wider_face_train_bbx_gt.txt or wider_face_val_bbx_gt.txt

    Returns:
        Dictionary mapping image filenames to lists of [x, y, w, h] bounding boxes
    """
    annotations: Dict[str, List[List[int]]] = {}

    with open(anno_path, 'r') as f:
        lines: List[str] = f.readlines()

    i: int = 0
    while i < len(lines):
        filename: str = lines[i].strip()
        i += 1

        if i >= len(lines):
            break

        num_faces_str: str = lines[i].strip()
        if not num_faces_str.isdigit():
            i += 1
            continue

        num_faces: int = int(num_faces_str)
        i += 1

        bboxes: List[List[int]] = []
        for _ in range(num_faces):
            if i >= len(lines):
                break

            parts: List[str] = lines[i].strip().split()

            if len(parts) >= 10:
                x: int = int(parts[0])
                y: int = int(parts[1])
                w: int = int(parts[2])
                h: int = int(parts[3])

                # Filter invalid boxes (too small or invalid)
                if w > 10 and h > 10 and x >= 0 and y >= 0:
                    bboxes.append([x, y, w, h])

            i += 1

        if bboxes:
            annotations[filename] = bboxes

    return annotations


def convert_split(
    img_dir: Path,
    anno_file: Path,
    out_img_dir: Path,
    out_label_dir: Path,
    split_name: str
) -> int:
    """
    Convert a single Wider Face split to YOLO format.

    Args:
        img_dir: Directory containing Wider Face images (with event subdirectories)
        anno_file: Path to annotation .txt file
        out_img_dir: Output directory for images
        out_label_dir: Output directory for YOLO labels
        split_name: Name of the split ('train' or 'val')

    Returns:
        Number of images converted
    """
    print(f"  Converting {split_name} split...")

    annotations: Dict[str, List[List[int]]] = parse_wider_annotation(str(anno_file))
    print(f"    Found {len(annotations)} annotated images in annotation file")

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    count: int = 0

    for subdir in img_dir.iterdir():
        if not subdir.is_dir():
            continue

        for img_file in subdir.glob('*.jpg'):
            rel_path: str = f"{subdir.name}/{img_file.name}"

            if rel_path not in annotations:
                continue

            img = cv2.imread(str(img_file))
            if img is None:
                print(f"    WARNING: Cannot read {img_file}")
                continue

            h, w = img.shape[:2]

            # Save image with unique name
            out_img_name: str = f"{subdir.name}_{img_file.name}"
            out_img_path: Path = out_img_dir / out_img_name
            cv2.imwrite(str(out_img_path), img)

            # Convert bounding boxes to YOLO format
            yolo_lines: List[str] = []
            for box in annotations[rel_path]:
                x, y, bw, bh = box

                # YOLO format: class_id cx cy w h (all normalized)
                cx: float = (x + bw / 2.0) / w
                cy: float = (y + bh / 2.0) / h
                nw: float = bw / w
                nh: float = bh / h

                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # Save label file
            label_name: str = f"{subdir.name}_{img_file.stem}.txt"
            label_path: Path = out_label_dir / label_name
            with open(str(label_path), 'w') as f:
                f.write('\n'.join(yolo_lines))

            count += 1

    print(f"    Converted {count} images for {split_name} split")
    return count


def convert_widerface_to_yolo(
    widerface_root: str,
    output_dir: str,
    splits: Optional[List[str]] = None
) -> None:
    """
    Convert Wider Face dataset to YOLO format.

    Expected input structure:
        widerface_root/
        ├── WIDER_train/images/
        ├── WIDER_val/images/
        └── wider_face_split/
            ├── wider_face_train_bbx_gt.txt
            └── wider_face_val_bbx_gt.txt

    Output structure:
        output_dir/
        ├── images/train/
        ├── images/val/
        ├── labels/train/
        └── labels/val/

    Args:
        widerface_root: Path to Wider Face root directory
        output_dir: Output directory for YOLO format dataset
        splits: List of splits to convert (default: ['train', 'val'])
    """
    if splits is None:
        splits = ['train', 'val']

    root: Path = Path(widerface_root)
    output: Path = Path(output_dir)

    if not root.exists():
        raise FileNotFoundError(f"Wider Face root not found: {root}")

    print(f"Converting Wider Face from: {root}")
    print(f"Output directory: {output}")
    print()

    total: int = 0

    for split in splits:
        if split == 'train':
            img_dir: Path = root / 'WIDER_train' / 'images'
            anno_file: Path = root / 'wider_face_split' / 'wider_face_train_bbx_gt.txt'
        elif split == 'val':
            img_dir = root / 'WIDER_val' / 'images'
            anno_file = root / 'wider_face_split' / 'wider_face_val_bbx_gt.txt'
        else:
            print(f"  Unknown split: {split}, skipping")
            continue

        if not img_dir.exists():
            print(f"  ERROR: Image directory not found: {img_dir}")
            continue

        if not anno_file.exists():
            print(f"  ERROR: Annotation file not found: {anno_file}")
            continue

        out_img_dir: Path = output / 'images' / split
        out_label_dir: Path = output / 'labels' / split

        count: int = convert_split(img_dir, anno_file, out_img_dir, out_label_dir, split)
        total += count

    print(f"\nTotal images converted: {total}")
    print(f"Dataset ready at: {output}")


def verify_dataset(data_yaml: str) -> bool:
    """
    Verify YOLO dataset integrity.

    Args:
        data_yaml: Path to dataset YAML file

    Returns:
        True if dataset is valid
    """
    import yaml

    with open(data_yaml, 'r') as f:
        data: Dict = yaml.safe_load(f)

    base_path: Path = Path(str(data['path']))

    if not base_path.exists():
        print(f"ERROR: Base path does not exist: {base_path}")
        return False

    all_valid: bool = True

    for split in ['train', 'val']:
        img_dir: Path = base_path / str(data[split])
        label_dir: Path = base_path / str(data[split]).replace('images', 'labels')

        if not img_dir.exists():
            print(f"ERROR: Image directory not found: {img_dir}")
            all_valid = False
            continue

        images: List[Path] = (
            list(img_dir.glob('*.jpg')) +
            list(img_dir.glob('*.jpeg')) +
            list(img_dir.glob('*.png'))
        )

        labels: List[Path] = []
        if label_dir.exists():
            labels = list(label_dir.glob('*.txt'))

        print(f"  {split}: {len(images)} images, {len(labels)} labels")

        if len(images) == 0:
            print(f"  ERROR: No images found in {img_dir}")
            all_valid = False
            continue

        # Check image-label correspondence
        if labels:
            img_stems: set = {img.stem for img in images}
            label_stems: set = {label.stem for label in labels}

            missing_labels: set = img_stems - label_stems
            extra_labels: set = label_stems - img_stems

            if missing_labels:
                print(f"  WARNING: {len(missing_labels)} images without labels")
            if extra_labels:
                print(f"  WARNING: {len(extra_labels)} labels without images")

        # Check a few labels for valid format
        if labels:
            sample_label: Path = labels[0]
            with open(str(sample_label), 'r') as f:
                first_line: str = f.readline().strip()
            parts: List[str] = first_line.split()
            if len(parts) != 5:
                print(f"  WARNING: Label format may be invalid. Expected 5 values per line.")
                print(f"  Sample: {first_line}")

    if all_valid:
        print("\nDataset verification passed.")
    else:
        print("\nDataset verification FAILED.")

    return all_valid


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Wider Face dataset to YOLO format"
    )
    parser.add_argument(
        '--widerface_root', type=str, required=True,
        help='Path to Wider Face root directory'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./datasets/widerface',
        help='Output directory for YOLO format dataset'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify dataset after conversion'
    )

    args = parser.parse_args()

    # Convert
    convert_widerface_to_yolo(args.widerface_root, args.output_dir)

    # Verify
    if args.verify:
        print("\n" + "="*60)
        verify_dataset('./widerface.yaml')
#!/usr/bin/env python3
"""Assignment 3-2: compare SSD-Mobilenet-v2 detections on two still images.
Run this script on the Jetson with jetson-inference Python bindings installed.
"""
import argparse
import csv
from pathlib import Path
import jetson.inference
import jetson.utils


def main():
    parser = argparse.ArgumentParser(description='Assignment 3-2 still-image object detection')
    parser.add_argument('--image', required=True, help='Path to an input JPG/PNG image')
    parser.add_argument('--threshold', type=float, required=True, help='Confidence threshold, e.g. 0.3 or 0.7')
    parser.add_argument('--output-dir', default='results', help='Where results are saved')
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error('--threshold must be between 0 and 1')
    input_path = Path(args.image).expanduser().resolve()
    if not input_path.is_file():
        parser.error('Image not found: {}'.format(input_path))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_tag = str(args.threshold).replace('.', 'p')
    stem = '{}_t{}'.format(input_path.stem, threshold_tag)

    net = jetson.inference.detectNet('ssd-mobilenet-v2', threshold=args.threshold)
    img = jetson.utils.loadImage(str(input_path))
    detections = net.Detect(img)
    rows = []
    print('IMAGE: {}'.format(input_path))
    print('THRESHOLD: {:.2f}'.format(args.threshold))
    print('NUMBER OF DETECTED OBJECTS: {}'.format(len(detections)))
    for i, d in enumerate(detections, start=1):
        class_name = net.GetClassDesc(d.ClassID)
        row = {
            'Object': i, 'ClassID': d.ClassID, 'ClassName': class_name,
            'Confidence': round(float(d.Confidence), 6),
            'Left': round(float(d.Left), 3), 'Top': round(float(d.Top), 3),
            'Right': round(float(d.Right), 3), 'Bottom': round(float(d.Bottom), 3),
            'Width': round(float(d.Width), 3), 'Height': round(float(d.Height), 3),
            'Area': round(float(d.Area), 3),
            'Center': '({:.3f}, {:.3f})'.format(float(d.Center[0]), float(d.Center[1])),
        }
        rows.append(row)
        print('OBJECT {}: {}'.format(i, row))
    image_out = output_dir / (stem + '_detected.jpg')
    csv_out = output_dir / (stem + '_detections.csv')
    log_out = output_dir / (stem + '_summary.txt')
    jetson.utils.saveImage(str(image_out), img)
    headers = ['Object', 'ClassID', 'ClassName', 'Confidence', 'Left', 'Top', 'Right', 'Bottom', 'Width', 'Height', 'Area', 'Center']
    with open(str(csv_out), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    with open(str(log_out), 'w', encoding='utf-8') as f:
        f.write('Input: {}\nThreshold: {}\nCount: {}\n'.format(input_path, args.threshold, len(rows)))
        for row in rows:
            f.write(str(row) + '\n')
    print('SAVED IMAGE: {}'.format(image_out))
    print('SAVED CSV: {}'.format(csv_out))
    print('SAVED SUMMARY: {}'.format(log_out))


if __name__ == '__main__':
    main()

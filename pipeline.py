from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFont
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import pandas as pd
import fastdeploy as fd
import cv2
import os
import numpy as np
import easyocr
import pandas as pd
from PIL import Image
det_model = fd.vision.detection.YOLOv8('stamp_layout_rotation.onnx')
det_model.preprocessor.size = [800, 800]

def extract_easyocr(image_path, mode=['LINE']):
    reader = easyocr.Reader(['en'], gpu=False)  # Initialize EasyOCR with English on CPU
    results = reader.readtext(image_path, detail=1)  # detail=1 for bounding box and confidence

    img = Image.open(image_path)
    width, height = img.size
    data = []
    
    # Process results
    for result in results:
        # Each result in EasyOCR output with detail=1 is (box, text, confidence)
        if result:
            box = result[0]
            text = result[1]
            confidence = result[2]

            # Extract all x and y coordinates from the box
            x_coords = [int(point[0]) for point in box]
            y_coords = [int(point[1]) for point in box]

            # Find the minimum and maximum x and y coordinates
            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)

            # Store data in a dictionary and append to list
            data.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'text': text,
                'confidence': confidence
            })

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2', 'text', 'confidence'])
    return df
def get_yolo_boxes(image_path, dump_boxes=False):
    im = cv2.imread(image_path)
    h, w, _ = im.shape  # Get the dimensions of the image
    result = det_model.predict(im)

    names = [
        "box", "table", "column", "header",
        "signature", "figure", "paragraph", "logo", "kv", "stamp"
    ]
    
    # Mapping from label names to IDs (0-indexed)
    name_to_id = {name: i for i, name in enumerate(names)}

    df_data = []
    yolo_data = []

    for i in range(len(result.boxes)):
        xmin, ymin, xmax, ymax = result.boxes[i]
        label_id = result.label_ids[i]
        confidence = result.scores[i]
        if confidence > .25:
            df_data.append({
                "x1": int(xmin),
                "y1": int(ymin),
                "x2": int(xmax),
                "y2": int(ymax),
                "label": names[label_id],  # Use label name
                "confidence": confidence
            })
            if dump_boxes:
                # Calculate normalized values
                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h
                # Append as list to ensure correct data types
                yolo_data.append([name_to_id[names[label_id]], x_center, y_center, width, height])
    
    # Convert list of data to DataFrame
    df = pd.DataFrame(df_data, columns=['x1', 'y1', 'x2', 'y2', 'label', 'confidence'])

    if dump_boxes:
        # Convert YOLO data to DataFrame for consistent formatting
        yolo_df = pd.DataFrame(yolo_data, columns=['class_id', 'x_center', 'y_center', 'width', 'height'])
        # Ensure data types are correctly set: class_id as int, others as float
        yolo_df = yolo_df.astype({'class_id': int, 'x_center': float, 'y_center': float, 'width': float, 'height': float})
        txt_path = image_path.rsplit('.', 1)[0] + '.txt'  # Supports multiple image extensions
        # Dump using pandas to_csv for consistent formatting, ensure no index and header, and specify float format
        yolo_df.to_csv(txt_path, sep=' ', index=False, header=False)
    
    return df  # Return the path to the .txt file for confirmation


def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def inside_iou(det_box, parent_box, return_iou=False):
    xi1 = max(det_box[0], parent_box[0])
    yi1 = max(det_box[1], parent_box[1])
    xi2 = min(det_box[2], parent_box[2])
    yi2 = min(det_box[3], parent_box[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
    iou = inter_area / float(det_area) if det_area != 0 else 0
    if return_iou:
        return iou
    else:
        return iou > 0.5

def filter_boundingbox_output(df_textract):
    # Convert DataFrame rows to a list of dictionaries for easier manipulation
    textract_data = df_textract.to_dict('records')
    filtered_boxes = []
    skip_indices = set()
    for i, box_a in enumerate(textract_data):
        if i in skip_indices:  # Skip this box if it's already marked as contained within another
            continue
        for j, box_b in enumerate(textract_data):
            if i != j and j not in skip_indices:
                # Check if box_b is completely inside box_a
                if inside_iou([box_b['x1'], box_b['y1'], box_b['x2'], box_b['y2']], 
                              [box_a['x1'], box_a['y1'], box_a['x2'], box_a['y2']], return_iou=True) == 1:
                    # Mark box_b to be skipped in future iterations
                    skip_indices.add(j)
        # After checking all boxes for containment, add the current box to the filtered list
        filtered_boxes.append(box_a)
    # Convert the filtered list of boxes back to a DataFrame
    df_textract_filtered = pd.DataFrame(filtered_boxes)
    return df_textract_filtered

def calculate_composite_dimensions(boxes, horizontal_padding, vertical_buffer, font_size, text_width=100, gap = 5):
    """
    Calculate the dimensions for the composite image, including additional space for text.
    """
    # Load the font and calculate text size for a representative string
    font = ImageFont.truetype("arial.ttf", font_size)
    text_height = max(textsize(f"{box[0]}, {box[1]}, {box[2]}, {box[3]}", font=font)[1] for box in boxes)
    
    max_padded_width = max(box[2] - box[0] for box in boxes) + 2 * horizontal_padding + text_width
    total_height = sum(box[3] - box[1] + text_height + gap for box in boxes)
    
    return max_padded_width, total_height

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height
def get_median_crop_color(crops):
    # Convert crops to grayscale and flatten the list of pixel values
    pixels = np.concatenate([np.array(crop.convert('L')).flatten() for crop in crops])
    # Calculate the median pixel value
    median_pixel = int(np.median(pixels))
    return (median_pixel, median_pixel, median_pixel)  # Return as RGB for compatibility
def parse_extracted_text(extracted_text):
    rows = []
    last_valid_coords = None

    for line in extracted_text:
        if ':' in line:
            coordinates_part, text_part = line.split(':', 1)
        else:
            # Handle case with no colon (e.g., text continuation)
            coordinates_part, text_part = '', line

        # Clean and split the coordinates part
        coord_strs = coordinates_part.replace('.', ',').strip().split(',')
        text_str = text_part.strip()

        # Try parsing coordinates
        try:
            coords = [int(coord.strip()) for coord in coord_strs if coord.strip()]
            if len(coords) == 4:
                last_valid_coords = {'x1': coords[0], 'y1': coords[1], 'x2': coords[2], 'y2': coords[3], 'text': text_str}
                rows.append(last_valid_coords)
            else:
                raise ValueError("Not enough valid coordinates")
        except ValueError:
            # Handle invalid or missing coordinates
            if last_valid_coords and text_str:
                # If there's pending text without coordinates, append it to the last valid entry's text
                last_valid_coords['text'] += ' ' + text_str

    return pd.DataFrame(rows)
def crop_and_extract_text(image_path, boxes, gap=0, horizontal_padding=10, vertical_buffer=10, minimal_gap=5, target_text_height=30):
    original_img = Image.open(image_path)
    crops = [original_img.crop((box[0], box[1], box[2], box[3])) for box in boxes]
    median_color = get_median_crop_color(crops)
    font = ImageFont.truetype("arial.ttf", target_text_height)  # Use target_text_height as an initial guess
    fontwidth, fontheight = textsize('100, 100, 100, 100 :', font=font)
    # Prepare the composite image
    max_width, total_height = calculate_composite_dimensions(boxes, horizontal_padding, vertical_buffer, target_text_height, fontwidth)
    composite_img = Image.new('RGB', (max_width, total_height), median_color)
    draw = ImageDraw.Draw(composite_img)
    current_y = vertical_buffer
    for i, (box, crop) in enumerate(zip(boxes, crops)):
        enhancer = ImageEnhance.Contrast(crop)
        enhanced_img = enhancer.enhance(2)
        padded_img = ImageOps.expand(enhanced_img, border=(horizontal_padding, vertical_buffer), fill=median_color)
        crop_width, crop_height = padded_img.size
        # Text preparation
        text = f"{box[0]}, {box[1]}, {box[2]}, {box[3]} :"
        text_size = textsize(text, font=font)
        text_x = minimal_gap
        crop_y_centroid = current_y + crop_height // 2
        text_y = crop_y_centroid - text_size[1] // 2
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        crop_x = text_size[0] + minimal_gap
        composite_img.paste(padded_img.convert('L'), (crop_x, current_y))
        current_y += crop_height + gap
    temp_path = 'temp_composite.jpg'
    composite_img.save(temp_path)
    # OCR to extract text including coordinates
    df_text = extract_easyocr(temp_path, mode=['LINE'])
    extracted_text = df_text['text'].tolist()
    parsed_text = parse_extracted_text(extracted_text)
    return pd.DataFrame(parsed_text)

def match_and_draw(image_path, df_textract, df_yolo_orig, mode='draw'):
    df_textract = filter_boundingbox_output(df_textract)
    relevant_classes = ['box', 'kv', 'signature', 'header']
    df_yolo = filter_boundingbox_output(df_yolo_orig[df_yolo_orig['label'].isin(relevant_classes)])
    df_yolo_table = filter_boundingbox_output(df_yolo_orig[df_yolo_orig['label'].isin(['table', 'column'])])
    combined_data = []
    matched_yolo_indices = set()  # Use a set for efficient look-up
    unmatched_textract_indices = set(df_textract.index)  # Initialize with all textract indices
    unmatched_yolo_boxes = []
    raw_image = cv2.imread(image_path)
    
    if mode == 'draw':
        original_image = raw_image.copy()
        textract_image = raw_image.copy()
        yolo_image = raw_image.copy()
        
        for _, row in df_textract.iterrows():
            cv2.rectangle(textract_image, (row['x1'], row['y1']), (row['x2'], row['y2']), (0, 0, 255), 2)
        
        for _, row in df_yolo.iterrows():
            if row['label'] in relevant_classes:
                cv2.rectangle(yolo_image, (row['x1'], row['y1']), (row['x2'], row['y2']), (255, 0, 0), 2)
        
        for _, row in df_yolo_table.iterrows():
            cv2.rectangle(original_image, (row['x1'], row['y1']), (row['x2'], row['y2']), (125, 0, 125), 3)
    
    for index_y, row_y in df_yolo.iterrows():
        max_iou = 0
        best_match_index_t = None
        for index_t, row_t in df_textract.iterrows():
            iou = get_iou([row_t['x1'], row_t['y1'], row_t['x2'], row_t['y2']], 
                          [row_y['x1'], row_y['y1'], row_y['x2'], row_y['y2']])
            if iou > max_iou:
                max_iou = iou
                best_match_index_t = index_t

        if max_iou > 0:
            matched_yolo_indices.add(index_y)
            if best_match_index_t is not None:
                unmatched_textract_indices.discard(best_match_index_t)
                combined_data.append({
                    "yolo_coords": [row_y['x1'], row_y['y1'], row_y['x2'], row_y['y2']],
                    "textract_coords": [[df_textract.loc[best_match_index_t][col] for col in ['x1', 'y1', 'x2', 'y2']]],
                    "textract_texts": [df_textract.loc[best_match_index_t]['text']]
                })
            # Draw green boxes for matched YOLO boxes
            if mode == 'draw':
                cv2.rectangle(original_image, (row_y['x1'], row_y['y1']), (row_y['x2'], row_y['y2']), (0, 255, 0), 2)
    # Draw red boxes for unmatched Textract boxes
    for index_t in unmatched_textract_indices:
        row_t = df_textract.loc[index_t]
        combined_data.append({
            "yolo_coords": [],
            "textract_coords": [[row_t['x1'], row_t['y1'], row_t['x2'], row_t['y2']]],
            "textract_texts": [row_t['text']]
        })
        if mode == 'draw':
            cv2.rectangle(original_image, (row_t['x1'], row_t['y1']), (row_t['x2'], row_t['y2']), (0, 0, 255), 2)

    # Draw Blue boxes for unmatched YOLO boxes of relevant classes
    for index_y, row_y in df_yolo.iterrows():
        if index_y not in matched_yolo_indices and row_y['label'] in relevant_classes:
            unmatched_yolo_boxes.append([row_y['x1'], row_y['y1'], row_y['x2'], row_y['y2']])
            if mode == 'draw':
                cv2.rectangle(original_image, (row_y['x1'], row_y['y1']), (row_y['x2'], row_y['y2']), (255, 0, 0), 2)
    
    missed_boxes = []
    for box in unmatched_yolo_boxes:
        # Modify extract_text_from_image to suit this specific use case
        missed_boxes.append(box)
    if missed_boxes:
        # Call the function to extract text for the missed boxes
        df_missed_text = crop_and_extract_text(image_path, missed_boxes)
        # Re-match the newly extracted texts with the YOLO boxes based on IOU
        for index_y, row_y in df_yolo.iterrows():
            if row_y['label'] in relevant_classes and index_y not in matched_yolo_indices:
                max_iou = 0
                best_match_index_t = None
                for index_t, row_t in df_missed_text.iterrows():
                    iou = get_iou([row_t['x1'], row_t['y1'], row_t['x2'], row_t['y2']], 
                                [row_y['x1'], row_y['y1'], row_y['x2'], row_y['y2']])
                    #print([row_t['x1'], row_t['y1'], row_t['x2'], row_t['y2']], [row_y['x1'], row_y['y1'], row_y['x2'], row_y['y2']], iou)
                    if iou > max_iou:
                        max_iou = iou
                        best_match_index_t = index_t

                if max_iou > 0:
                    matched_yolo_indices.add(index_y)
                    combined_data.append({
                        "yolo_coords": [row_y['x1'], row_y['y1'], row_y['x2'], row_y['y2']],
                        "textract_coords": [[df_missed_text.loc[best_match_index_t][col] for col in ['x1', 'y1', 'x2', 'y2']]],
                        "textract_texts": [df_missed_text.loc[best_match_index_t]['text']]
                    })

    if mode == 'draw':
        for data in combined_data:
            # Each entry in combined_data should have 'textract_coords' and 'textract_texts'
            for coords, text in zip(data['textract_coords'], data['textract_texts']):
                # Ensuring coords is a list with 4 elements [x1, y1, x2, y2]
                if isinstance(coords, list) and len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    position = (max(x1 - 10, 0), max((y1+y2)//2, 0))
                    cv2.putText(original_image, text, position, cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 125, 125), 1)
        h_min = min(textract_image.shape[0], yolo_image.shape[0], original_image.shape[0])
        textract_image = cv2.resize(textract_image, (int(textract_image.shape[1] * h_min / textract_image.shape[0]), h_min))
        yolo_image = cv2.resize(yolo_image, (int(yolo_image.shape[1] * h_min / yolo_image.shape[0]), h_min))
        original_image = cv2.resize(original_image, (int(original_image.shape[1] * h_min / original_image.shape[0]), h_min))
        final_image = cv2.hconcat([textract_image, yolo_image, original_image])

        # Save the combined image
        output_image_path = 'results/' + os.path.basename(image_path).replace('.jpg', 'combined.jpg')
        cv2.imwrite(output_image_path, final_image)
    return pd.DataFrame(combined_data)


import glob
from tqdm import tqdm
for image_path in tqdm(glob.glob('rectified/*.jpg')):
    df_textract = extract_easyocr(image_path)
    df_yolo = get_yolo_boxes(image_path)
    match_and_draw(image_path, df_textract, df_yolo)

import cv2

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def line_intersection(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # Calculate the intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    return px, py

def is_point_inside_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

def calculate_mask_area(mask):
    area = cv2.contourArea(mask)
    return area

def associate_boxes_with_tracks(boxes, tracks, clases, masks, intersection_line, count_id):
    trackerMasks = []
    intersection_counter = 0

    for xmin, ymin, xmax, ymax, track_id in tracks:
        best_iou = 0
        best_box_index = -1

        for i in range(len(boxes)):
            iou = calculate_iou([xmin, ymin, xmax, ymax], boxes[i])

            if iou > best_iou:
                best_iou = iou
                best_box_index = i

        if best_iou > 0:
            trackerMasks.append([xmin, ymin, xmax, ymax, track_id, clases[best_box_index], masks[best_box_index]])

            # Verificar intersección con la línea
            intersection_point = line_intersection([intersection_line[0], intersection_line[1]], [(xmin, ymin), (xmax, ymax)])
            if is_point_inside_box(intersection_point, [xmin, ymin, xmax, ymax]) and track_id not in count_id:
                area = calculate_mask_area(masks[best_box_index])
                intersection_counter += area
                count_id.append(track_id)


    return trackerMasks, intersection_counter
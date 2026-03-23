import json
import re


def parse_coordinates(content: str, image_shape: tuple) -> list[int]:
    """Parse coordinates from VLM response. Returns [x, y] center point."""
    h, w = image_shape[:2]

    # Try direct array format [[x1, y1, x2, y2]]
    try:
        data = json.loads(content)
        if isinstance(data, list) and len(data) > 0:
            coords = data[0] if isinstance(data[0], list) else data
            if len(coords) == 4:
                x_center = (coords[0] + coords[2]) / 2
                y_center = (coords[1] + coords[3]) / 2
                return [int(x_center * w / 1000), int(y_center * h / 1000)]
    except:
        pass

    # Try JSON format first
    json_match = re.search(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(1))
        if data:
            if isinstance(data[0], dict):
                if "point_2d" in data[0]:
                    point = data[0]["point_2d"]
                    return [int(point[0] * w / 1000), int(point[1] * h / 1000)]
                elif "bbox_2d" in data[0]:
                    bbox = data[0]["bbox_2d"]
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    return [int(x_center * w / 1000), int(y_center * h / 1000)]
            elif isinstance(data[0], (int, float)) and len(data) == 4:
                # Flat bbox array [x1, y1, x2, y2]
                x_center = (data[0] + data[2]) / 2
                y_center = (data[1] + data[3]) / 2
                return [int(x_center * w / 1000), int(y_center * h / 1000)]

    # Try (x, y) format
    match = re.search(r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)', content)
    if match:
        x, y = float(match.group(1)), float(match.group(2))
        if x <= 1.0 and y <= 1.0:
            x, y = int(x * w), int(y * h)
        return [int(x), int(y)]

    raise ValueError(f"Could not parse coordinates from: {content}")

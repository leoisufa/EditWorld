"""Glass-style WASD and camera-rotation overlay for inference videos."""

import cv2
import numpy as np


def extract_controls(c2ws, translation_threshold=0.002, rotation_threshold=0.001):
    """Classify dominant camera-relative translation and rotation per frame."""
    relative = np.linalg.inv(c2ws[:-1]) @ c2ws[1:]
    wasd = np.zeros((len(c2ws), 4), dtype=np.uint8)
    rotations = ["none"] * len(c2ws)
    for index, transform in enumerate(relative):
        right, _, forward = transform[:3, 3]
        if max(abs(right), abs(forward)) >= translation_threshold:
            if abs(forward) > abs(right):
                wasd[index, 0 if forward > 0 else 2] = 1
            else:
                wasd[index, 3 if right > 0 else 1] = 1
        rotation = transform[:3, :3]
        sy = np.hypot(rotation[0, 0], rotation[1, 0])
        if sy < 1e-6:
            yaw = np.arctan2(-rotation[2, 0], rotation[0, 0])
            pitch = np.arctan2(-rotation[1, 2], rotation[1, 1])
        else:
            yaw = np.arctan2(-rotation[2, 0], sy)
            pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
        if max(abs(yaw), abs(pitch)) >= rotation_threshold:
            if abs(yaw) >= abs(pitch):
                rotations[index] = "right" if yaw > 0 else "left"
            else:
                rotations[index] = "up" if pitch > 0 else "down"
    if len(c2ws) > 1:
        wasd[-1] = wasd[-2]
        rotations[-1] = rotations[-2]
    return wasd, rotations


def rounded_rect(image, p1, p2, color, radius, alpha=1.0, border=None):
    x1, y1 = p1
    x2, y2 = p2
    layer = image.copy()
    cv2.rectangle(layer, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(layer, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for center in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(layer, center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0, image)
    if border is not None:
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), border, 1, cv2.LINE_AA)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), border, 1, cv2.LINE_AA)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), border, 1, cv2.LINE_AA)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), border, 1, cv2.LINE_AA)
        for center, start, end in (
            ((x1 + radius, y1 + radius), 180, 270),
            ((x2 - radius, y1 + radius), 270, 360),
            ((x2 - radius, y2 - radius), 0, 90),
            ((x1 + radius, y2 - radius), 90, 180),
        ):
            cv2.ellipse(image, center, (radius, radius), 0, start, end, border, 1, cv2.LINE_AA)


def glass_panel(frame, p1, p2, radius):
    x1, y1 = p1
    x2, y2 = p2
    roi = frame[y1:y2, x1:x2]
    if roi.size:
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (15, 15), 0)
    rounded_rect(frame, p1, p2, (18, 22, 30), radius, alpha=0.72, border=(110, 120, 135))


def draw_key(frame, x, y, size, label, active, accent, scale):
    shadow = max(2, round(3 * scale))
    radius = max(4, round(9 * scale))
    rounded_rect(frame, (x + shadow, y + shadow), (x + size + shadow, y + size + shadow),
                 (0, 0, 0), radius, alpha=0.35)
    color = accent if active else (48, 53, 63)
    border = tuple(min(255, value + 45) for value in color) if active else (100, 108, 120)
    rounded_rect(frame, (x, y), (x + size, y + size), color, radius,
                 alpha=0.94 if active else 0.78, border=border)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.60 * scale
    thickness = max(1, round(1.6 * scale))
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    origin = (x + (size - text_size[0]) // 2, y + (size + text_size[1]) // 2)
    cv2.putText(frame, label, origin, font, font_scale, (250, 250, 250), thickness, cv2.LINE_AA)


def draw_arrow_key(frame, x, y, size, direction, active, scale):
    draw_key(frame, x, y, size, "", active, (35, 145, 245), scale)
    cx, cy = x + size // 2, y + size // 2
    span = round(size * 0.20)
    if direction == "up":
        points = [(cx - span, cy + span // 2), (cx, cy - span), (cx + span, cy + span // 2)]
    elif direction == "down":
        points = [(cx - span, cy - span // 2), (cx, cy + span), (cx + span, cy - span // 2)]
    elif direction == "left":
        points = [(cx + span // 2, cy - span), (cx - span, cy), (cx + span // 2, cy + span)]
    else:
        points = [(cx - span // 2, cy - span), (cx + span, cy), (cx - span // 2, cy + span)]
    cv2.polylines(frame, [np.asarray(points, np.int32)], False, (255, 255, 255),
                  max(2, round(2.5 * scale)), cv2.LINE_AA)


def draw_controls(frame, wasd, rotation):
    height, width = frame.shape[:2]
    scale = max(min(width / 832.0, height / 480.0), 0.55)
    key = round(38 * scale)
    gap = round(6 * scale)
    pad = round(12 * scale)
    margin = round(24 * scale)
    panel_w = 3 * key + 2 * gap + 2 * pad
    panel_h = 2 * key + gap + 2 * pad
    radius = round(14 * scale)

    left_x, top_y = margin, height - margin - panel_h
    glass_panel(frame, (left_x, top_y), (left_x + panel_w, top_y + panel_h), radius)
    key_x, key_y = left_x + pad, top_y + pad
    positions = {
        "W": (key_x + key + gap, key_y, 0),
        "A": (key_x, key_y + key + gap, 1),
        "S": (key_x + key + gap, key_y + key + gap, 2),
        "D": (key_x + 2 * (key + gap), key_y + key + gap, 3),
    }
    for label, (x, y, index) in positions.items():
        draw_key(frame, x, y, key, label, bool(wasd[index]), (205, 125, 25), scale)

    right_x = width - margin - panel_w
    glass_panel(frame, (right_x, top_y), (right_x + panel_w, top_y + panel_h), radius)
    key_x = right_x + pad
    arrows = {
        "up": (key_x + key + gap, key_y),
        "left": (key_x, key_y + key + gap),
        "down": (key_x + key + gap, key_y + key + gap),
        "right": (key_x + 2 * (key + gap), key_y + key + gap),
    }
    for direction, (x, y) in arrows.items():
        draw_arrow_key(frame, x, y, key, direction, rotation == direction, scale)
    return frame


def visualize_wasd_and_rotation_ui(frames, c2ws, wasd_actions=None):
    """Draw the supplied UI directly on RGB float frames in [0, 1]."""
    if len(frames) != len(c2ws):
        raise ValueError(f"Frame/pose mismatch: frames={len(frames)}, poses={len(c2ws)}")
    pose_wasd, rotations = extract_controls(np.asarray(c2ws))
    wasd = pose_wasd if wasd_actions is None else np.asarray(wasd_actions)
    if len(wasd) != len(frames):
        raise ValueError(f"Frame/action mismatch: frames={len(frames)}, actions={len(wasd)}")
    if wasd_actions is not None and len(wasd) > 1:
        wasd = wasd.copy()
        wasd[-1] = wasd[-2]
    output = np.ascontiguousarray((np.clip(frames, 0, 1) * 255).astype(np.uint8)[..., ::-1])
    for index in range(len(output)):
        draw_controls(output[index], wasd[index], rotations[index])
    return output[..., ::-1].astype(np.float32) / 255.0

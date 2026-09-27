"""Minimal WASD and camera-rotation visualization for inference outputs."""

import cv2
import numpy as np


REFERENCE_UI_WIDTH = 832.0
REFERENCE_UI_HEIGHT = 480.0


def _ui_scale(width, height):
    return max(min(width / REFERENCE_UI_WIDTH, height / REFERENCE_UI_HEIGHT), 0.35)


def _scaled(value, scale, minimum=1):
    return max(minimum, int(round(value * scale)))


def _draw_box(frame, x, y, size, color, radius, alpha=0.7):
    overlay = frame.copy()
    x2, y2 = x + size, y + size
    cv2.rectangle(overlay, (x + radius, y), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x + radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_chevron(frame, center, size, direction, thickness):
    x, y = center
    half = size // 2
    points = {
        "up": ((x - half, y + half // 2), (x, y - half // 2), (x + half, y + half // 2)),
        "down": ((x - half, y - half // 2), (x, y + half // 2), (x + half, y - half // 2)),
        "left": ((x + half // 2, y - half), (x - half // 2, y), (x + half // 2, y + half)),
        "right": ((x - half // 2, y - half), (x + half // 2, y), (x - half // 2, y + half)),
    }[direction]
    cv2.polylines(
        frame,
        [np.asarray(points, dtype=np.int32)],
        isClosed=False,
        color=(255, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_wasd(frame, action, x, y, scale):
    size = _scaled(40, scale, 12)
    gap = _scaled(5, scale, 2)
    radius = _scaled(5, scale, 2)
    thickness = _scaled(2, scale)
    font_scale = max(0.35, 0.6 * scale)
    keys = {
        "W": (x + size + gap, y, 0),
        "A": (x, y + size + gap, 1),
        "S": (x + size + gap, y + size + gap, 2),
        "D": (x + 2 * (size + gap), y + size + gap, 3),
    }
    for label, (kx, ky, index) in keys.items():
        color = (0, 100, 200) if action[index] > 0.5 else (50, 50, 50)
        _draw_box(frame, kx, ky, size, color, radius)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = kx + (size - text_size[0]) // 2
        text_y = ky + (size + text_size[1]) // 2
        cv2.putText(
            frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )


def _draw_directions(frame, active_direction, width, height, scale):
    size = _scaled(40, scale, 12)
    gap = _scaled(5, scale, 2)
    radius = _scaled(5, scale, 2)
    margin = _scaled(50, scale, 16)
    x = width - margin - 3 * size - 2 * gap
    y = height - margin - 2 * size - gap
    buttons = {
        "up": (x + size + gap, y),
        "left": (x, y + size + gap),
        "down": (x + size + gap, y + size + gap),
        "right": (x + 2 * (size + gap), y + size + gap),
    }
    for direction, (bx, by) in buttons.items():
        color = (200, 100, 0) if direction == active_direction else (50, 50, 50)
        _draw_box(frame, bx, by, size, color, radius)
        _draw_chevron(
            frame,
            (bx + size // 2, by + size // 2),
            _scaled(size * 0.45, 1.0, 7),
            direction,
            _scaled(3, scale),
        )


def _rotation_directions(c2ws, threshold):
    relative = np.matmul(np.linalg.inv(c2ws[:-1]), c2ws[1:])
    directions = []
    for rotation in relative[:, :3, :3]:
        sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
        if sy < 1e-6:
            yaw = np.arctan2(-rotation[2, 0], rotation[0, 0])
            pitch = np.arctan2(-rotation[1, 2], rotation[1, 1])
        else:
            yaw = np.arctan2(-rotation[2, 0], sy)
            pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
        if max(abs(yaw), abs(pitch)) < threshold:
            directions.append("none")
        elif abs(yaw) >= abs(pitch):
            directions.append("right" if yaw > 0 else "left")
        else:
            directions.append("up" if pitch > 0 else "down")
    return directions + ["none"]


def _translation_wasd(c2ws, threshold):
    relative = np.matmul(np.linalg.inv(c2ws[:-1]), c2ws[1:])
    actions = []
    for transform in relative:
        right, _, forward = transform[:3, -1]
        action = [0, 0, 0, 0]
        if max(abs(right), abs(forward)) >= threshold:
            if abs(forward) > abs(right):
                action[0 if forward > 0 else 2] = 1
            else:
                action[3 if right > 0 else 1] = 1
        actions.append(action)
    return np.asarray(actions + [[0, 0, 0, 0]])


def visualize_wasd_and_rotation_ui(
    frames,
    c2ws=None,
    wasd_actions=None,
    translation_threshold=0.01,
    rotation_threshold=0.005,
):
    """Overlay only WASD movement keys and up/down/left/right rotation arrows."""
    if c2ws is None:
        raise ValueError("c2ws is required for rotation visualization")
    if wasd_actions is None:
        wasd_actions = _translation_wasd(c2ws, translation_threshold)
    rotations = _rotation_directions(c2ws, rotation_threshold)

    output = (frames * 255).astype(np.uint8)[..., ::-1]
    frame_count, height, width, _ = output.shape
    scale = _ui_scale(width, height)
    key_size = _scaled(40, scale, 12)
    gap = _scaled(5, scale, 2)
    x = _scaled(50, scale, 16)
    y = height - _scaled(65, scale, 20) - 2 * key_size - gap

    for index in range(min(frame_count, len(wasd_actions), len(rotations))):
        _draw_wasd(output[index], wasd_actions[index], x, y, scale)
        _draw_directions(output[index], rotations[index], width, height, scale)

    return output[..., ::-1].astype(np.float32) / 255.0

import cv2


class InputManager:
    def __init__(self):
        self.clicked = None
        self.buttons = {}

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x1, y1, x2, y2) in self.buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.clicked = name
                break

    def set_buttons(self, buttons):
        self.buttons = buttons

    def consume_click(self):
        clicked = self.clicked
        self.clicked = None
        return clicked


def handle_text_input(current: str, key: int, max_len: int = 32) -> str:
    """Update a text field string given a waitKey result.

    - Printable ASCII (32-126) appends the character.
    - Backspace (8) removes the last character.
    Returns the updated string unchanged for any other key.
    """
    if key == 8 and current:          # backspace
        return current[:-1]
    if 32 <= key <= 126 and len(current) < max_len:
        return current + chr(key)
    return current

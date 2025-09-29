import os
import sys
import random
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from sam2.build_sam import build_sam2_video_predictor  # type: ignore


class SAM2GUI:
    """
    A GUI class for interacting with SAM2 video segmentation.
    Provides an interactive interface for annotating objects across video frames.
    """

    def __init__(
        self,
        video_dir=None,
        tmp_dir="tmp",
        checkpoint_path="../sam2/checkpoints/sam2.1_hiera_large.pt",
        config_path="../sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    ):
        """
        Initialize the SAM2 GUI.

        Args:
            video_dir (str): Path to directory containing video frames
            output_mask_dir (str): Path to directory for saving masks
            checkpoint_path (str): Path to SAM2 checkpoint file
            config_path (str): Path to SAM2 config file
        """
        # Set default paths if not provided
        self.video_dir = video_dir
        self.temp_dir = os.path.abspath(tmp_dir)
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        else:
            # Clear temp directory
            for f in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, f))

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load images
        self._prepare_images()
        self._load_images()

        # Initialize SAM2
        self._load_sam2()

        # GUI State
        self.current_frame = 0
        self.clicks = {}  # obj_id -> list of (x, y, label)
        self.current_obj_id = 1
        self.video_segments = {}

        # GUI components
        self.fig = None
        self.ax = None
        self.buttons = {}
        self.text_boxes = {}

        # Setup GUI
        self._setup_gui()

    def _prepare_images(self):
        """Convert images to JPG under input_root."""
        self.mapping = {}
        counter = 0

        for root, dirs, files in os.walk(self.video_dir):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")):
                    new_name = f"{counter:05d}.jpg"
                    new_path = os.path.join(self.temp_dir, new_name)

                    # Convert to JPG and save
                    with Image.open(os.path.join(root, file)) as im:
                        if im.mode in ("RGBA", "LA") or (
                            im.mode == "P" and "transparency" in im.info
                        ):
                            # Create white background image
                            background = Image.new("RGB", im.size, (255, 255, 255))
                            background.paste(
                                im.convert("RGBA"), mask=im.convert("RGBA").split()[-1]
                            )
                            background.save(new_path, "JPEG")
                        else:
                            rgb_im = im.convert("RGB")
                            rgb_im.save(new_path, "JPEG")

                    self.mapping[new_name] = {
                        "original_path": root,
                        "original_name": file,
                    }
                    counter += 1

    def _load_images(self):
        """Load and sort video frames."""
        self.frame_names = sorted(
            [
                f
                for f in os.listdir(self.temp_dir)
                if f.lower().endswith((".jpg", ".jpeg"))
            ],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        self.frames = [
            Image.open(os.path.join(self.temp_dir, f)) for f in self.frame_names
        ]
        self.num_frames = len(self.frames)

    def _load_sam2(self):
        """Load SAM2 predictor and initialize inference state."""
        print("Loading SAM2...")
        self.predictor = build_sam2_video_predictor(
            self.config_path, self.checkpoint_path, device=self.device
        )
        self.inference_state = self.predictor.init_state(video_path=self.temp_dir)
        self.predictor.reset_state(self.inference_state)

    def _setup_gui(self):
        """Setup the matplotlib GUI interface."""
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.38)  # More space for 2 rows

        # Connect mouse events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Setup buttons and controls
        self._create_buttons()

    def _create_buttons(self):
        """Create all GUI buttons and text boxes."""
        # Row 1: Main Actions
        self.buttons["segment"] = Button(plt.axes([0.01, 0.22, 0.1, 0.05]), "Segment")
        self.buttons["segment"].on_clicked(lambda e: self.segment_current())

        self.buttons["propagate"] = Button(
            plt.axes([0.12, 0.22, 0.15, 0.05]), "Propagate"
        )
        self.buttons["propagate"].on_clicked(self.propagate_all)

        self.buttons["save"] = Button(plt.axes([0.28, 0.22, 0.1, 0.05]), "Save")
        self.buttons["save"].on_clicked(self.save_masks)

        self.buttons["clear"] = Button(
            plt.axes([0.39, 0.22, 0.12, 0.05]), "Clear Clicks"
        )
        self.buttons["clear"].on_clicked(self.clear_clicks)

        self.buttons["restart"] = Button(plt.axes([0.52, 0.22, 0.1, 0.05]), "Restart")
        self.buttons["restart"].on_clicked(self.restart)

        # Row 2: Navigation + ID + Jump
        self.buttons["prev"] = Button(plt.axes([0.01, 0.12, 0.1, 0.05]), "Prev")
        self.buttons["prev"].on_clicked(self.prev_frame)

        self.buttons["next"] = Button(plt.axes([0.12, 0.12, 0.1, 0.05]), "Next")
        self.buttons["next"].on_clicked(self.next_frame)

        self.text_boxes["obj_id"] = TextBox(
            plt.axes([0.23, 0.12, 0.12, 0.05]),
            "Obj ID",
            initial=str(self.current_obj_id),
        )
        self.text_boxes["obj_id"].on_submit(self.set_obj_id)

        self.text_boxes["jump"] = TextBox(
            plt.axes([0.36, 0.12, 0.12, 0.05]),
            "Jump to",
            initial=str(self.current_frame),
        )
        self.text_boxes["jump"].on_submit(self.jump_to_frame)

        self.buttons["jump"] = Button(plt.axes([0.49, 0.12, 0.1, 0.05]), "Jump")
        self.buttons["jump"].on_clicked(
            lambda e: self.jump_to_frame(self.text_boxes["jump"].text)
        )

    def update_display(self):
        """Update the display with current frame and annotations."""
        self.ax.clear()
        self.ax.imshow(self.frames[self.current_frame])

        # Show clicks
        for obj_id, pts in self.clicks.items():
            for x, y, label in pts:
                color = "green" if label == 1 else "red"
                self.ax.scatter(x, y, c=color, s=100, marker="*")

        self.ax.set_title(f"Frame {self.current_frame} | Obj ID: {self.current_obj_id}")
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Handle mouse click events for annotation."""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Determine label from mouse button
        if event.button == 1:  # Left click = positive
            label = 1
            print(f"Positive click at ({x:.1f}, {y:.1f})")
        elif event.button == 3:  # Right click = negative
            label = 0
            print(f"Negative click at ({x:.1f}, {y:.1f})")
        else:
            return  # Ignore other buttons

        self.clicks.setdefault(self.current_obj_id, []).append((x, y, label))
        self.update_display()

    def clear_clicks(self, event=None):
        """Clear all annotation clicks."""
        self.clicks = {}
        self.update_display()

    def segment_current(self):
        """Segment objects in current frame based on clicks."""
        print("Segmenting...")
        for obj_id, pts in self.clicks.items():
            coords = np.array([[x, y] for x, y, _ in pts], dtype=np.float32)
            labels = np.array([label for _, _, label in pts], dtype=np.int32)

            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=self.current_frame,
                obj_id=obj_id,
                points=coords,
                labels=labels,
            )

            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0).cpu().numpy()
                colored_mask = mask.astype(np.float32) * out_obj_id
                self.ax.imshow(colored_mask.squeeze(), alpha=0.5, cmap="jet")
                print(f"Displayed mask for obj_id {out_obj_id}")

        self.fig.canvas.draw_idle()

    def propagate_all(self, event=None):
        """Propagate masks to all frames in the video."""
        print("Propagating masklets...")
        self.video_segments = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        print("Propagation done.")
        self.update_display()

    def generate_color_map(self, num_classes):
        """Generate a color map for visualizing instance masks."""
        random.seed(42)  # For reproducibility
        color_map = {0: (0, 0, 0)}  # background = black
        for i in range(1, num_classes + 1):
            color_map[i] = tuple(random.randint(0, 255) for _ in range(3))
        return color_map

    def save_masks(self, event=None):
        """Save masks as .npy and colored .png files."""
        print("Saving masks as .npy and colored .png files...")

        for frame_idx, masks in self.video_segments.items():
            # Create an empty mask
            first_mask = next(iter(masks.values()))
            mask_shape = np.squeeze(first_mask).shape
            full_mask = np.zeros(mask_shape, dtype=np.uint8)

            object_ids = list(masks.keys())

            for obj_id, mask in masks.items():
                mask = np.squeeze(mask)
                if mask.ndim != 2:
                    raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
                full_mask[mask > 0] = obj_id

            frame_name = self.frame_names[frame_idx]
            original_path = self.mapping[frame_name]["original_path"]
            original_name = self.mapping[frame_name]["original_name"]
            output_mask_dir = os.path.join(original_path, "masks")
            if not os.path.exists(output_mask_dir):
                os.makedirs(output_mask_dir)

            # Save .npy file
            npy_filename = f"{original_name}.npy"
            npy_path = os.path.join(output_mask_dir, npy_filename)
            np.save(npy_path, full_mask)

            # Generate and save color visualization
            color_map = self.generate_color_map(max(object_ids))
            png_mask = np.zeros((*mask_shape, 3), dtype=np.uint8)

            for obj_id, color in color_map.items():
                png_mask[full_mask == obj_id] = color

            png_filename = f"{original_name}.png"
            png_path = os.path.join(output_mask_dir, png_filename)
            cv2.imwrite(png_path, png_mask)

            print(f"Saved: {npy_path}, {png_path}")

    def next_frame(self, event=None):
        """Navigate to next frame."""
        self.current_frame = min(self.current_frame + 1, self.num_frames - 1)
        self.update_display()

    def prev_frame(self, event=None):
        """Navigate to previous frame."""
        self.current_frame = max(self.current_frame - 1, 0)
        self.update_display()

    def set_obj_id(self, text):
        """Set current object ID."""
        try:
            self.current_obj_id = int(text)
            self.update_display()
        except ValueError:
            print("Invalid object ID")

    def jump_to_frame(self, text):
        """Jump to specific frame."""
        try:
            idx = int(text)
            if 0 <= idx < self.num_frames:
                self.current_frame = idx
                self.update_display()
            else:
                print(f"Frame {idx} out of range (0–{self.num_frames - 1})")
        except ValueError:
            print("Invalid frame index")

    def restart(self, event=None):
        """Restart the annotation session."""
        print("Restarting session...")

        # Clear all segmentation data and clicks
        self.video_segments.clear()
        self.clicks.clear()

        # Reset frame and object ID
        self.current_frame = 0
        self.current_obj_id = 1

        # Reset text box value
        self.text_boxes["obj_id"].set_val(str(self.current_obj_id))

        # Update display
        self.update_display()

        print("Session restarted.")

    def run(self):
        """Start the GUI application."""
        print("GUI ready.")
        print("Left-click = Positive (green), Right-click = Negative (red)")
        self.update_display()
        plt.show()


if __name__ == "__main__":
    # Create and run the SAM2 GUI
    video_dir = input("Enter path to video frames directory: ")
    gui = SAM2GUI(
        video_dir=video_dir,
    )
    gui.run()

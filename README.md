# SAM2NeRF

A simple GUI tool for video segmentation using SAM2.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dougefla/sam2nerf
   cd sam2nerf
   ```

2. Run the installation script:
   ```bash
   ./install.sh
   ```

## Usage

1. Make sure you are in the `/workspace` directory

2. Run the GUI application:
   ```bash
   python sam2_gui.py
   ```

3. Input your video frame directory when prompted

4. In the GUI:
   - **Right-click** and **left-click** to interact with frames
   - **Segment 1-5 frames** as needed
   - **Propagate and save** your results

That's it! The tool will help you segment objects across video frames using SAM2.
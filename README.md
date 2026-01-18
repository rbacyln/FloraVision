# 🌸 FloraVision - AR Gesture Filter

**FloraVision** is a real-time Augmented Reality (AR) application that uses computer vision to overlay interactive digital elements based on hand gestures and face detection. Built with Python, OpenCV, and MediaPipe.

## ✨ Features

The application detects the number of fingers raised and triggers specific AR overlays:

* 🦋 **1 Finger:** Summons a **Butterfly** on the fingertip.
* 🐦 **2 Fingers:** Calls a **Bird** to the hand.
* 💐 **3 Fingers:** Places a **Flower Bouquet** in the hand.
* 👑 **5 Fingers + Face:** Detects the face and places a **Royal Crown** perfectly on the head.

## 🛠️ Tech Stack

* **Python 3.x**
* **OpenCV:** For image processing and video capture.
* **MediaPipe:** For robust Hand and Face landmark detection.
* **NumPy:** For matrix operations.

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/rbacyln/FloraVision.git](https://github.com/rbacyln/FloraVision.git)
    cd FloraVision
    ```

2.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the App:**
    ```bash
    python main.py
    ```

## 📸 Usage

1.  Run the script. The camera window will open automatically (positioned at the top right).
2.  Show your hand to the camera.
3.  Try different numbers of fingers (1, 2, 3, or 5) to see the magic!
4.  Press **'q'** or click the close button to exit.

## 📂 Project Structure

FloraVision/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── butterfly.png
├── bird.png
├── bouquet.png
└── crown.png

## 📄 License

This project is open-source and available under the MIT License.
# AirCanvas AI — Intelligent Gesture-Based Digital Canvas 🎨🤖

### 🖐️ Real-Time Hand Gesture Controlled Virtual Drawing System

**Air Canvas AI Pro** is a real-time **Computer Vision + Machine Learning** application that allows users to draw and interact with a virtual canvas using only **hand gestures** captured through a webcam. 🖥️📷

The system uses **MediaPipe** for hand landmark detection, **OpenCV** for real-time image processing and rendering, and **NumPy** for numerical operations. Hand movements are converted into digital strokes, while different finger configurations are used to control drawing tools and interact with the virtual toolbar. 🧠🖐️

The project demonstrates how **AI-powered perception + Computer Vision + gesture recognition** can be combined to create a natural, touch-free human-computer interaction system. 🚀

---

# ✨ Features

### 🖐️ Hand Tracking

* 🎯 Real-time hand detection
* 🔢 21-point hand landmark tracking
* 📍 Accurate fingertip tracking
* 🔄 Continuous hand movement tracking
* 🧠 Finger-state analysis using hand geometry

### ✏️ Drawing

* ☝️ Index finger drawing
* 🖌️ Smooth digital strokes
* 🎨 Multiple colors
* 📏 Adjustable brush size
* 🌊 Adaptive movement smoothing
* 🚫 Unwanted movement-jump filtering

### 🎨 Air Toolbar

Control the application without touching the keyboard or mouse.

```text
🧹 CLEAR
🔵 BLUE
🟢 GREEN
🔴 RED
🟡 YELLOW
🧽 ERASE
↩️ UNDO
↪️ REDO
💾 SAVE
🔴 POINTER
```

### 🧠 Gesture Interaction

| 🖐️ Gesture                | ⚡ Action                 |
| -------------------------- | ------------------------ |
| ☝️ Index finger only       | ✏️ Draw                  |
| ☝️🖕 Index + middle finger | 🎨 Select toolbar        |
| 🤏 Thumb + index pinch     | 📏 Change brush size     |
| 🖐️ Four fingers           | 🖥️ Presentation pointer |

---

# 🎨 Air Color Selection

One of the main features is **touch-free color selection**.

### Example

```text
       ☝️🖕
         │
         ▼
   ┌─────────────┐
   │ 🟢 GREEN    │
   └─────────────┘
         │
      Hold briefly
         │
         ▼
    🟢 GREEN ON
         │
         ▼
        ☝️
         │
         ▼
       ✏️ DRAW
```

The user raises the **index + middle fingers**, moves over a toolbar button, and holds briefly.

The application uses **selection stability frames + cooldown/debounce logic** to reduce accidental clicks. 🎯🛡️

---

# 🧹 Air Eraser

The eraser can be activated without touching the keyboard.

### Air Method

```text
☝️🖕
 ↓
🧽 ERASE
 ↓
ERASER ON
 ↓
☝️
 ↓
Move over drawing
 ↓
🧹 Erased
```

### Keyboard

Press:

```text
E
```

to toggle the eraser.

---

# 🧠 System Architecture

```text
                    📷 WEBCAM
                       │
                       ▼
                🔵 OpenCV Capture
                       │
                       ▼
              🖼️ Frame Processing
                       │
                       ▼
             🧠 MediaPipe Hands
                       │
                       ▼
                🔢 21 Landmarks
                       │
                       ▼
              🖐️ Finger Analysis
                       │
                       ▼
              🎯 Gesture Recognition
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
     ✏️ DRAW        🎨 SELECT      🤏 PINCH
        │              │              │
        ▼              ▼              ▼
   🌊 Smoothing    🛡️ Debounce    📏 Brush Size
        │              │
        ▼              ▼
   🖌️ Canvas       🧰 Toolbar
        │              │
        └───────┬──────┘
                ▼
          🖼️ Virtual Canvas
                │
                ▼
             💾 PNG
```

---

# 🔬 Computer Vision Pipeline

## 1️⃣ Webcam Capture 📷

OpenCV captures frames from the webcam and horizontally flips the image to provide a natural mirror-style interaction.

## 2️⃣ Hand Landmark Detection 🖐️

MediaPipe detects the hand and provides **21 landmark coordinates** representing important points of the hand.

## 3️⃣ Finger-State Classification 🧠

The application analyzes the geometric relationships between finger joints to determine which fingers are extended.

```text
☝️ Index
   +
🖕 Middle
   ↓
🎨 SELECT
```

```text
☝️ Index
   ↓
✏️ DRAW
```

```text
🤏 Thumb + Index
   ↓
📏 BRUSH SIZE
```

## 4️⃣ Fingertip Tracking 🎯

The index fingertip acts as the primary interaction point.

Its position is continuously tracked and converted into canvas coordinates.

## 5️⃣ Motion Stabilization 🌊

Natural hand movement and webcam noise can cause fingertip jitter.

The project therefore applies **adaptive temporal smoothing** to produce smoother movement.

```text
Raw Hand Position
       ↓
🌊 Adaptive Smoothing
       ↓
🎯 Stable Position
       ↓
✏️ Smooth Stroke
```

## 6️⃣ Movement-Jump Filtering 🛡️

If the detected fingertip suddenly jumps an unrealistic distance between frames, the movement is rejected.

This helps prevent:

```text
❌ Random long lines
❌ Broken strokes
❌ Tracking glitches
```

and produces:

```text
✅ Cleaner drawing
✅ More stable movement
✅ Better user experience
```

## 7️⃣ Gesture Debouncing 🎯

Toolbar actions require the gesture to remain stable for several frames before an action is triggered.

This prevents accidental:

```text
❌ Color switching
❌ Unwanted clearing
❌ Accidental tool activation
```

---

# ✏️ Drawing Engine

When the application detects the **index-finger-only gesture**, the fingertip becomes the virtual brush.

```text
🖐️ Hand
 ↓
🎯 Index Fingertip
 ↓
🌊 Adaptive Smoothing
 ↓
🛡️ Jump Validation
 ↓
📍 Coordinate Tracking
 ↓
🖌️ OpenCV Rendering
 ↓
🎨 Virtual Canvas
```

The drawing canvas is maintained separately from the webcam frame, allowing the drawing to remain visible while the live camera feed continues running. 🖥️

---

# 🧰 Available Tools

### 🎨 Colors

* 🔵 Blue
* 🟢 Green
* 🔴 Red
* 🟡 Yellow

### 🧽 Eraser

Erase existing strokes using the virtual eraser.

### ↩️ Undo

Restore the previous canvas state.

### ↪️ Redo

Restore an undone canvas state.

### 📏 Brush Size

Change brush size using:

* 🤏 Pinch gesture
* ⌨️ Keyboard controls

### 🔺 Shape Recognition

The project includes basic recognition for:

* ➖ Lines
* ▭ Rectangles
* ⭕ Circles
* 🔺 Triangles

### 🖥️ Presentation Pointer

Use hand gestures as a virtual presentation pointer.

---

# 🎮 Keyboard Controls

| ⌨️ Key | ⚡ Action          |
| ------ | ----------------- |
| `1`    | 🔵 Blue           |
| `2`    | 🟢 Green          |
| `3`    | 🔴 Red            |
| `4`    | 🟡 Yellow         |
| `C`    | 🧹 Clear          |
| `S`    | 💾 Save           |
| `Z`    | ↩️ Undo           |
| `Y`    | ↪️ Redo           |
| `E`    | 🧽 Eraser         |
| `P`    | 🖥️ Pointer       |
| `T`    | 🔺 Shape mode     |
| `[`    | 📉 Decrease brush |
| `]`    | 📈 Increase brush |
| `Q`    | 🚪 Quit           |
| `ESC`  | 🚪 Quit           |

---

# 🎯 Accuracy & Stability

The project contains multiple mechanisms designed to improve real-time tracking and interaction stability. 🛡️

### 🔍 Detection

* 🎯 High hand-detection confidence
* 🎯 High hand-tracking confidence
* 🖐️ 21-point landmark tracking

### 🌊 Movement

* Adaptive temporal smoothing
* Fingertip stabilization
* Movement-jump filtering
* Missing-frame handling

### 🎨 Interaction

* Gesture stability threshold
* Selection cooldown
* Debounce logic
* Toolbar exclusion area

### 🖌️ Rendering

* Anti-aliased strokes
* Continuous coordinate tracking
* Separate drawing canvas

---

# 🛠️ Technologies Used

| Technology                 | Purpose                     |
| -------------------------- | --------------------------- |
| 🐍 **Python**              | Application development     |
| 👁️ **OpenCV**             | Computer Vision & rendering |
| 🖐️ **MediaPipe**          | Hand landmark detection     |
| 🔢 **NumPy**               | Numerical processing        |
| 🧠 **Machine Learning**    | Hand perception             |
| 🎯 **Gesture Recognition** | User interaction            |

---

# 📦 Installation

### 1️⃣ Install Python

Python **3.10 or 3.11** is recommended.

### 2️⃣ Install dependencies

```bash
python -m pip install --upgrade pip
```

```bash
python -m pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
python air_canvas_ai_pro.py
```

You can also open the Python file using **Python IDLE** and press:

```text
F5
```

---

# 💻 Recommended Environment

For better hand tracking performance:

* 💡 Use bright, even lighting
* 📷 Use a 720p or higher webcam
* 🖐️ Keep your complete hand inside the frame
* 🎥 Keep the camera stable
* 🧱 Avoid highly cluttered backgrounds
* 🌑 Avoid strong backlighting
* 📏 Maintain a reasonable distance from the camera

---

# 📁 Project Structure

```text
Air_Canvas_AI_Pro/
│
├── 🐍 air_canvas_ai_pro.py
├── 📦 requirements.txt
├── 📖 README.md
│
├── 🎞️ gif/
│   └── air_canvas_ai_demo.gif
│
├── 📚 original_reference/
│   ├── Air-canvas.py
│   └── air_canvas_ml.py
│
└── 🖼️ output/
    └── saved drawings
```

---

# 🧠 Machine Learning Component

Air Canvas AI Pro uses a **pre-trained MediaPipe hand landmark model** for hand perception.

The project does not train a custom neural network from scratch. Instead, the ML model provides the hand landmarks while the application implements the Computer Vision and interaction pipeline around them.

```text
🧠 Pre-trained ML Model
          ↓
🖐️ 21 Hand Landmarks
          ↓
📐 Finger Geometry
          ↓
🎯 Gesture Recognition
          ↓
🌊 Temporal Filtering
          ↓
🖌️ Interactive Canvas
```

This demonstrates how a pre-trained ML perception model can be integrated into a practical real-time Computer Vision application.

---

# 🌍 Applications

The same technology can be extended to:

* 🖌️ Touchless drawing systems
* 🧑‍🏫 Interactive digital whiteboards
* 🖥️ Presentation control
* 🎮 Gesture-based interfaces
* ♿ Accessibility-oriented interfaces
* 🏫 Educational applications
* 🏢 Touch-free kiosks
* 🎨 Virtual painting
* 🤝 Human-computer interaction research

---

# 🔮 Future Improvements

Possible future versions can include:

* 🧠 Custom neural-network gesture classifier
* ✍️ Advanced handwriting recognition
* 🔢 Handwritten digit classification
* 🖐️ Multi-hand drawing
* 🤖 AI-powered sketch recognition
* ✨ Automatic shape correction
* 🗣️ Offline voice recognition
* 🖥️ Gesture-controlled presentations
* 🌐 Web-based deployment
* 🤝 Real-time collaborative drawing
* 🎨 AI sketch-to-image generation
* 🧠 Custom gesture dataset and model training

---

# 💼 Resume Project Description

### 🎨 Air Canvas AI Pro — Real-Time Hand Gesture Controlled Computer Vision System

Developed a real-time virtual drawing application using **Python, OpenCV, MediaPipe, and NumPy**. Implemented **21-point hand landmark tracking, geometric finger-state classification, adaptive temporal smoothing, gesture debouncing, movement-jump filtering, air-based tool selection, erasing, undo/redo, pinch-controlled brush sizing, shape recognition, PNG export, and presentation-pointer interaction.**

---

# 📊 Project Highlights

This project demonstrates practical experience in:

* 🧠 Machine Learning model integration
* 👁️ Real-time Computer Vision
* 🖐️ Hand landmark detection
* 🎯 Gesture recognition
* 🤝 Human-computer interaction
* 🖼️ Image processing
* 📍 Coordinate transformation
* 🌊 Temporal filtering
* ⚡ Real-time application development
* 🎨 Interactive UI design

---

# 🔐 Optional Features

### 📝 Handwriting OCR

The project contains an optional OCR integration point for converting handwritten content into text.

### 🗣️ Voice Commands

An optional voice-command integration can be enabled using the additional speech-recognition dependencies.

These features remain optional so that the core Air Canvas system can run without them. ⚙️

---

# 🚀 Getting Started

```text
1️⃣ Install Python
       ↓
2️⃣ Install requirements
       ↓
3️⃣ Connect webcam
       ↓
4️⃣ Run air_canvas_ai_pro.py
       ↓
5️⃣ Show your hand 🖐️
       ↓
6️⃣ Select a color 🎨
       ↓
7️⃣ Raise index finger ☝️
       ↓
8️⃣ Start drawing ✏️
       ↓
9️⃣ Save your artwork 💾
```

---

# ⭐ Why This Project?

Air Canvas AI Pro demonstrates how **Machine Learning, Computer Vision, and gesture recognition** can be combined to create a natural and intuitive user interface.

Instead of relying on a traditional:

🖱️ Mouse
⌨️ Keyboard
🖥️ Touchscreen

the user can interact with the application using:

🖐️ **Their hands.**

This makes the project a practical demonstration of **touchless Human-Computer Interaction (HCI)** and real-time AI-powered interaction. 🚀🤖

---

# 📜 License

This project is intended for **educational, research, and portfolio purposes**.

Third-party libraries used by the project are subject to their respective licenses.

---

## 👨‍💻 Author

**Air Canvas AI Pro**

🎯 Computer Vision
🧠 Artificial Intelligence
🖐️ Hand Gesture Recognition
🤖 Machine Learning
🐍 Python
🚀 Real-Time AI Applications

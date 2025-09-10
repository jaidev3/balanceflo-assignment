# Enhanced Desk Edge Detection Application - Complete Analysis

This is an **AI-powered computer vision application** that detects desk edges in office images, even when people are present in the scene. Here's how it works:

## 🎯 **Main Purpose**

The app identifies the **user-facing edge of a desk** in photos, which is useful for:

- Augmented reality applications
- Desk measurement tools
- Office space analysis
- Camera positioning systems

## 🏗️ **Application Architecture**

### **1. Main Entry Point (`edge-detector.py`)**

This is a **Streamlit web application** that provides:

- **Image upload interface** (PNG/JPG/WEBP)
- **Interactive parameter controls** in the sidebar
- **Real-time visualization** of results
- **Debug views** for algorithm transparency

**Key UI Functions:**

- `st.file_uploader()` - Handles image uploads
- Parameter sliders for fine-tuning detection
- Debug checkboxes for algorithm visualization
- Download buttons for results (PNG + JSON)

### **2. Image Processing Pipeline (`image_utils.py`)**

**`preprocess_for_edges()`** - The core preprocessing function:

```python
def preprocess_for_edges(img_bgr, use_clahe=True, use_segmentation=False):
    # 1. Optional AI person/chair removal
    # 2. Convert to grayscale
    # 3. Optional CLAHE contrast enhancement
    # 4. Gaussian blur for noise reduction
    # 5. Canny edge detection
```

**Utility Functions:**

- `pil_to_cv()` - Converts PIL images to OpenCV format
- `cv_to_rgb()` - Converts back for Streamlit display
- `resize_max_width()` - Maintains aspect ratio while resizing

### **3. AI Enhancement Layer (`semantic_segmentation.py`)**

**`SemanticSegmentationMask` class** uses **DeepLabv3 ResNet-101**:

- **Purpose**: Remove people and chairs from images before edge detection
- **Model**: Pre-trained on COCO dataset
- **Classes**: Person (ID: 15) and Chair (ID: 62)
- **Process**:
  1. Resize image to 512x512
  2. Run semantic segmentation
  3. Create binary mask
  4. Remove masked areas (fill with black)

**Key Functions:**

- `get_person_chair_mask()` - Generates segmentation mask
- `apply_mask_to_image()` - Removes detected objects
- `get_cleaned_image()` - Complete pipeline

### **4. Dual Edge Detection System (`edge_detection_core.py`)**

The app uses **two complementary algorithms**:

#### **A. Straight Edge Detection (`detect_straight_edge()`)**

- **Algorithm**: Probabilistic Hough Transform
- **Purpose**: Finds linear desk edges
- **Process**:
  1. Focus on bottom 60% of image (user-facing zone)
  2. Apply Hough line detection
  3. Filter by angle tolerance (configurable)
  4. Score by length and position
- **Best for**: Rectangular desks, clear linear edges

#### **B. Curved Edge Detection (`detect_curved_edge()`)**

- **Algorithm**: Contour analysis with morphological operations
- **Purpose**: Finds curved or irregular desk edges
- **Process**:
  1. Focus on bottom portion of image
  2. Apply morphological closing/dilation
  3. Find contours
  4. Extract "user-facing envelope" (max Y for each X)
  5. Resample to fixed number of points
- **Best for**: Rounded desks, complex shapes

#### **C. Edge Selection (`choose_best_edge()`)**

- **Logic**: Compares both results and selects the best
- **Criteria**:
  - Length and prominence
  - Position in user-facing zone
  - Width coverage
  - Straight edges get slight preference if competitive

### **5. Data Models (`data_models.py`)**

**`EdgeResult` dataclass** stores detection results:

```python
@dataclass
class EdgeResult:
    edge_type: str        # "line" or "polyline"
    points: List[Tuple]   # Coordinate points
    score: float          # Confidence score
    method: str           # Algorithm used
    metadata: Dict        # Additional parameters
```

## 🔄 **Complete Application Flow**

1. **Image Upload** → User uploads office desk image
2. **Preprocessing** → Convert format, resize, optional AI cleaning
3. **Edge Detection** → Run both straight and curved algorithms in parallel
4. **Selection** → Choose best result based on scoring criteria
5. **Visualization** → Draw edge overlay on original image
6. **Output** → Generate JSON coordinates + download options
7. **Debug Views** → Show intermediate processing steps

## 🎛️ **Key Parameters (User Configurable)**

- **AI Enhancement**: Enable/disable person/chair removal
- **CLAHE**: Contrast enhancement for better edge detection
- **Min Line Length**: Minimum length for straight edges
- **Angle Tolerance**: How diagonal edges can be (10-80°)
- **User-Facing Zone**: Focus area (40-90% from bottom)
- **Contour Smoothing**: How much to smooth curved edges
- **Curve Detail**: Number of points in curved edges (8-200)

## 🔍 **Debug Features**

The app provides extensive debugging visualizations:

- **Grayscale view** - Preprocessed image
- **Canny edges** - Raw edge detection
- **Hough lines overlay** - Shows detected straight lines
- **Contour overlay** - Shows detected curved contours
- **AI segmentation** - Shows person/chair detection mask

## 💡 **Key Innovations**

1. **AI-Enhanced Preprocessing**: Removes people/chairs that would confuse edge detection
2. **Dual Algorithm Approach**: Handles both straight and curved desk edges
3. **User-Facing Focus**: Prioritizes edges closest to the camera/user
4. **Flexible Camera Angles**: Works with various viewing angles
5. **Real-time Tuning**: Interactive parameters for different scenarios

## 📊 **Application Flow Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI LAYER                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 1. Image Upload (PNG/JPG/WEBP)                             │ │
│  │ 2. Parameter Controls (Sidebar)                            │ │
│  │    - AI Enhancement (Person/Chair Removal)                 │ │
│  │    - Edge Detection Settings                               │ │
│  │    - Debug Visualization Options                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    IMAGE PREPROCESSING                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ image_utils.py: preprocess_for_edges()                     │ │
│  │ 1. Convert PIL → OpenCV BGR                                │ │
│  │ 2. Resize to max width (1280px)                            │ │
│  │ 3. Optional: AI Person/Chair Removal                       │ │
│  │ 4. Convert to Grayscale                                    │ │
│  │ 5. Optional: CLAHE Contrast Enhancement                    │ │
│  │ 6. Gaussian Blur                                           │ │
│  │ 7. Canny Edge Detection                                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI ENHANCEMENT LAYER                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ semantic_segmentation.py: remove_person_chair_noise()      │ │
│  │ 1. Load DeepLabv3 ResNet-101 Model (COCO Dataset)          │ │
│  │ 2. Segment Person (Class 15) & Chair (Class 62)            │ │
│  │ 3. Create Binary Mask                                      │ │
│  │ 4. Remove Masked Areas (Fill with Black)                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL EDGE DETECTION                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ edge_detection_core.py                                     │ │
│  │                                                             │ │
│  │ ┌─────────────────────┐    ┌─────────────────────────────┐  │ │
│  │ │ STRAIGHT EDGE       │    │ CURVED EDGE                  │  │ │
│  │ │ detect_straight_    │    │ detect_curved_edge()        │  │ │
│  │ │ edge()              │    │                             │  │ │
│  │ │                     │    │ 1. Focus on Bottom Zone     │  │ │
│  │ │ 1. Probabilistic    │    │ 2. Morphological Operations │  │ │
│  │ │    Hough Transform  │    │ 3. Find Contours            │  │ │
│  │ │ 2. Filter by Angle  │    │ 4. Extract User-Facing      │  │ │
│  │ │ 3. Score by Length  │    │    Envelope (Max Y by X)    │  │ │
│  │ │    & Position       │    │ 5. Resample Points          │  │ │
│  │ └─────────────────────┘    └─────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EDGE SELECTION                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ choose_best_edge()                                         │ │
│  │ 1. Compare Straight vs Curved Results                      │ │
│  │ 2. Apply User-Facing Bonuses                              │ │
│  │ 3. Prefer Straight if Competitive (85% threshold)          │ │
│  │ 4. Return Best EdgeResult                                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT GENERATION                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 1. draw_edge_annotation() - Visual Overlay                 │ │
│  │ 2. build_json() - Coordinate Export                        │ │
│  │ 3. Debug Visualizations (Optional)                         │ │
│  │ 4. Download Buttons (PNG + JSON)                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 **Key Components:**

### 1. **Main App (edge-detector.py)**

- Streamlit web interface
- Parameter controls and debug options
- Image upload and result display

### 2. **Image Processing (image_utils.py)**

- Format conversions (PIL ↔ OpenCV)
- Image resizing and preprocessing
- CLAHE contrast enhancement
- Canny edge detection

### 3. **AI Enhancement (semantic_segmentation.py)**

- DeepLabv3 ResNet-101 for person/chair detection
- Noise removal before edge detection
- Handles various camera angles

### 4. **Edge Detection Core (edge_detection_core.py)**

- **Straight Edge**: Hough Transform for linear desk edges
- **Curved Edge**: Contour analysis for curved desk edges
- **Selection Logic**: Chooses best result based on scoring

### 5. **Data Models (data_models.py)**

- EdgeResult dataclass for structured output
- Metadata tracking for debugging

## 📈 **Flow Summary:**

1. **Upload** → Image preprocessing with optional AI cleaning
2. **Detect** → Dual algorithm approach (straight + curved)
3. **Select** → Choose best result based on user-facing characteristics
4. **Output** → Visual annotation + JSON coordinates + debug views

This application is particularly robust because it combines traditional computer vision techniques with modern AI segmentation, making it work reliably even in cluttered office environments with people present.

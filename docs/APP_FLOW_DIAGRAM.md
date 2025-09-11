# Desk Edge Detection App - Flow Diagram

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

## Key Components:

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

## Flow Summary:

1. **Upload** → Image preprocessing with optional AI cleaning
2. **Detect** → Dual algorithm approach (straight + curved)
3. **Select** → Choose best result based on user-facing characteristics
4. **Output** → Visual annotation + JSON coordinates + debug views

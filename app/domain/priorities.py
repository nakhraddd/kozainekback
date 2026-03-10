# English names are used for keys as this is what the YOLO model provides.

HIGH_PRIORITY_OBJECTS = {
    "person",
    "car",
    "bicycle",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "traffic light",
    "stop sign",
    "stairs"
}

MEDIUM_PRIORITY_OBJECTS = {
    "chair",
    "bench",
    "dining table",
    "couch",
    "bed",
    "potted plant",
    "cat"
}

# Low priority objects are any objects not in the high or medium lists.

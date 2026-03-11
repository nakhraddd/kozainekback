from typing import List
from app.domain.models import ProcessedObject
from app.domain.priorities import HIGH_PRIORITY_OBJECTS

# This dictionary should be kept in sync with the one in detector.py
RUSSIAN_NAMES = {
    "person": "человек", "bicycle": "велосипед", "car": "машина", "motorcycle": "мотоцикл",
    "airplane": "самолет", "bus": "автобус", "train": "поезд", "truck": "грузовик",
    "boat": "лодка", "traffic light": "светофор", "fire hydrant": "пожарный гидрант",
    "stop sign": "знак стоп", "parking meter": "паркомат", "bench": "скамейка",
    "bird": "птица", "cat": "кошка", "dog": "собака", "horse": "лошадь", "sheep": "овца",
    "cow": "корова", "elephant": "слон", "bear": "медведь", "zebra": "зебра",
    "giraffe": "жираф", "backpack": "рюкзак", "umbrella": "зонт", "handbag": "сумка",
    "tie": "галстук", "suitcase": "чемодан", "frisbee": "фризби", "skis": "лыжи",
    "snowboard": "сноуборд", "sports ball": "спортивный мяч", "kite": "воздушный змей",
    "baseball bat": "бейсбольная бита", "baseball glove": "бейсбольная перчатка",
    "skateboard": "скейтборд", "surfboard": "доска для серфинга", "tennis racket": "теннисная ракетка",
    "bottle": "бутылка", "wine glass": "бокал", "cup": "чашка", "fork": "вилка",
    "knife": "нож", "spoon": "ложка", "bowl": "миска", "banana": "банан", "apple": "яблоко",
    "sandwich": "бутерброд", "orange": "апельсин", "broccoli": "брокколи", "carrot": "морковь",
    "hot dog": "хот-дог", "pizza": "пицца", "donut": "пончик", "cake": "торт",
    "chair": "стул", "couch": "диван", "potted plant": "растение в горшке", "bed": "кровать",
    "dining table": "обеденный стол", "toilet": "туалет", "tv": "телевизор", "laptop": "ноутбук",
    "mouse": "мышь", "remote": "пульт", "keyboard": "клавиатура", "cell phone": "телефон",
    "microwave": "микроволновка", "oven": "духовка", "toaster": "тостер", "sink": "раковина",
    "refrigerator": "холодильник", "book": "книга", "clock": "часы", "vase": "ваза",
    "scissors": "ножницы", "teddy bear": "плюшевый мишка", "hair drier": "фен",
    "toothbrush": "зубная щетка", "A4 paper": "бумага А4", "stairs": "лестница",
    "Caution! Possible obstacle ahead": "Внимание! Возможное припятсвие впереди"
}

def format_message(processed_objects: List[ProcessedObject]) -> str:
    if not processed_objects:
        return ""

    attention_prefix = ""
    # Check for the "Attention!" condition
    for p in processed_objects:
        # The original English name is needed to check against HIGH_PRIORITY_OBJECTS
        english_name = next((en for en, ru in RUSSIAN_NAMES.items() if ru == p.name), None)
        if english_name in HIGH_PRIORITY_OBJECTS and p.distance == "близко":
            attention_prefix = "Внимание! "
            break

    text_parts = []
    for p in processed_objects:
        distance_info = f"({p.distance_cm:.0f} см)" if p.distance_cm is not None else ""
        text_parts.append(f"{p.name} {p.distance} {p.position} {distance_info}".strip())
    
    return attention_prefix + ", ".join(text_parts)

from typing import List
from app.domain.models import ProcessedObject
from app.domain.priorities import HIGH_PRIORITY_OBJECTS

# Translations
TRANSLATIONS = {
    "RUSSIAN": {
        "names": {
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
            "Caution! Possible obstacle ahead": "Внимание! Возможное препятствие впереди"
        },
        "distance": {
            "close": "близко", "medium": "на среднем расстоянии", "far": "далеко"
        },
        "position": {
            "left": "слева", "center": "по центру", "right": "справа",
            "top": "сверху", "middle": "посередине", "bottom": "снизу"
        },
        "attention": "Внимание! ",
        "units": "см"
    },
    "ENGLISH": {
        "names": {},  # Defaults to English key if not found
        "distance": {
            "close": "close", "medium": "at medium distance", "far": "far away"
        },
        "position": {
            "left": "on the left", "center": "in the center", "right": "on the right",
            "top": "at the top", "middle": "in the middle", "bottom": "at the bottom"
        },
        "attention": "Attention! ",
        "units": "cm"
    },
    "KAZAKH": {
        "names": {
            "person": "адам", "bicycle": "велосипед", "car": "көлік", "motorcycle": "мотоцикл",
            "bus": "автобус", "truck": "жүк көлігі", "traffic light": "бағдаршам", "stop sign": "тоқта белгісі",
            "cat": "мысық", "dog": "ит", "chair": "орындық", "table": "үстел", "door": "есік",
            "stairs": "баспалдақ",
            "Caution! Possible obstacle ahead": "Назар аударыңыз! Алда кедергі болуы мүмкін"
        },
        "distance": {
            "close": "жақын", "medium": "орташа қашықтықта", "far": "алыс"
        },
        "position": {
            "left": "сол жақта", "center": "ортада", "right": "оң жақта",
            "top": "жоғарыда", "middle": "ортада", "bottom": "төменде"
        },
        "attention": "Назар аударыңыз! ",
        "units": "см"
    }
}

# Keep this for backward compatibility if needed, aliased to Russian
RUSSIAN_NAMES = TRANSLATIONS["RUSSIAN"]["names"]


def format_message(processed_objects: List[ProcessedObject], language: str = "RUSSIAN") -> str:
    if not processed_objects:
        return ""

    lang_data = TRANSLATIONS.get(language, TRANSLATIONS["RUSSIAN"])
    names_map = lang_data["names"]
    dist_map = lang_data["distance"]
    pos_map = lang_data["position"]

    attention_prefix = ""

    # 1. Determine Attention Prefix
    for p in processed_objects:
        key_name = p.name
        found_key = next((k for k, v in RUSSIAN_NAMES.items() if v == p.name), None)
        if found_key:
            key_name = found_key

        if key_name in HIGH_PRIORITY_OBJECTS:
            attention_prefix = lang_data["attention"]
            break

    text_parts = []
    for p in processed_objects:
        # Resolve Key Name
        key_name = p.name
        found_key = next((k for k, v in RUSSIAN_NAMES.items() if v == p.name), None)
        if found_key: key_name = found_key

        # Translate Name
        display_name = names_map.get(key_name, key_name)

        # Resolve Distance Key (Logic.py uses: "близко", "средне", "далеко")
        dist_key = "medium"
        if p.distance == "близко":
            dist_key = "close"
        elif p.distance == "далеко":
            dist_key = "far"
        # "средне" maps to default "medium"

        display_dist = dist_map.get(dist_key, p.distance)

        # Resolve Position Key (Logic.py uses: "слева", "по центру", "справа")
        pos_key = "center"
        if p.position == "слева":
            pos_key = "left"
        elif p.position == "справа":
            pos_key = "right"
        elif p.position == "сверху":
            pos_key = "top"
        elif p.position == "снизу":
            pos_key = "bottom"

        display_pos = pos_map.get(pos_key, p.position)

        distance_info = f"({p.distance_cm:.0f} {lang_data['units']})" if p.distance_cm is not None else ""

        # Construct phrase
        text_parts.append(f"{display_name} {display_dist} {display_pos} {distance_info}".strip())

    return attention_prefix + ", ".join(text_parts)
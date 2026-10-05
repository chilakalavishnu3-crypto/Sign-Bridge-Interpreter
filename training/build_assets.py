"""
SignBridge Asset Generator
Builds packs.json, templates.json, languages.json, regions.json, and audio index.
"""

import json
import os
import hashlib
from typing import Dict, Any

ASSETS_DIR = "assets"
PACKS_DIR = os.path.join(ASSETS_DIR, "packs")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")
CLIPS_DIR = os.path.join(ASSETS_DIR, "clips")


def build_languages():
    languages = [
        {"code": "en", "english_name": "English", "native_name": "English", "locale": "en-IN"},
        {"code": "ta", "english_name": "Tamil", "native_name": "தமிழ்", "locale": "ta-IN"},
        {"code": "hi", "english_name": "Hindi", "native_name": "हिन्दी", "locale": "hi-IN"},
        {"code": "te", "english_name": "Telugu", "native_name": "తెలుగు", "locale": "te-IN"},
        {"code": "kn", "english_name": "Kannada", "native_name": "ಕನ್ನಡ", "locale": "kn-IN"},
        {"code": "ml", "english_name": "Malayalam", "native_name": "മലയാളം", "locale": "ml-IN"}
    ]
    path = os.path.join(PACKS_DIR, "languages.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(languages, f, indent=2, ensure_ascii=False)
    print(f"Generated {path}")


def build_regions():
    regions = {
        "Tamil Nadu": {"primary": "ta", "common": ["ta", "en", "te", "kn"]},
        "Puducherry": {"primary": "ta", "common": ["ta", "en", "ml", "te"]},
        "Karnataka": {"primary": "kn", "common": ["kn", "en", "te", "ta"]},
        "Kerala": {"primary": "ml", "common": ["ml", "en", "ta", "hi"]},
        "Andhra Pradesh": {"primary": "te", "common": ["te", "en", "hi", "ta"]},
        "Telangana": {"primary": "te", "common": ["te", "hi", "en", "kn"]},
        "Delhi": {"primary": "hi", "common": ["hi", "en", "te", "ta"]},
        "Uttar Pradesh": {"primary": "hi", "common": ["hi", "en"]},
        "Bihar": {"primary": "hi", "common": ["hi", "en"]},
        "Madhya Pradesh": {"primary": "hi", "common": ["hi", "en"]},
        "Rajasthan": {"primary": "hi", "common": ["hi", "en"]},
        "Maharashtra": {"primary": "hi", "common": ["hi", "en", "te", "kn"]},
        "West Bengal": {"primary": "en", "common": ["en", "hi"]},
        "Gujarat": {"primary": "hi", "common": ["hi", "en"]}
    }
    path = os.path.join(PACKS_DIR, "regions.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2, ensure_ascii=False)
    print(f"Generated {path}")


def build_packs():
    packs = {
        "railway": {
            "name": "Railway Ticket Counter",
            "description": "Standard vocabulary for railway ticket counters and station inquiries",
            "signs": [
                "0", "1", "2", "3", "4", "5",
                "HELLO", "THANK YOU", "YES", "NO", "HELP",
                "WHERE", "WHEN", "WATER", "TOILET", "MONEY", "DOCTOR",
                "TICKET", "TRAIN", "PLATFORM"
            ],
            "quick_replies": [
                {
                    "id": "reply_platform_5",
                    "text": {
                        "en": "Platform 5",
                        "ta": "பிளாட்பார்ம் 5",
                        "hi": "प्लेटफॉर्म 5",
                        "te": "ప్లాట్‌ఫారమ్ 5",
                        "kn": "ಪ್ಲಾಟ್‌ಫಾರ್ಮ್ 5",
                        "ml": "പ്ലാറ്റ്‌ഫോം 5"
                    },
                    "clips": ["PLATFORM", "5"]
                },
                {
                    "id": "reply_200_rupees",
                    "text": {
                        "en": "200 rupees",
                        "ta": "200 ரூபாய்",
                        "hi": "200 रुपये",
                        "te": "200 రూపాయలు",
                        "kn": "200 ರೂಪಾಯಿ",
                        "ml": "200 രൂപ"
                    },
                    "clips": ["2", "0", "0", "MONEY"]
                },
                {
                    "id": "reply_train_4",
                    "text": {
                        "en": "Train at 4 o'clock",
                        "ta": "ரயில் 4 மணிக்கு வரும்",
                        "hi": "ट्रेन 4 बजे आएगी",
                        "te": "రైలు 4 గంటలకు వస్తుంది",
                        "kn": "ರೈಲು 4 ಗಂಟೆಗೆ ಬರುತ್ತದೆ",
                        "ml": "ട്രെയിൻ 4 മണിക്ക് എത്തും"
                    },
                    "clips": ["TRAIN", "4"]
                },
                {
                    "id": "reply_wait_here",
                    "text": {
                        "en": "Please wait here",
                        "ta": "இங்கே காத்திருங்கள்",
                        "hi": "कृपया यहाँ प्रतीक्षा करें",
                        "te": "దయచేసి ఇక్కడ వేచి ఉండండి",
                        "kn": "ದಯವಿಟ್ಟು ಇಲ್ಲಿ ನಿರೀಕ್ಷಿಸಿ",
                        "ml": "ദയവായി ഇവിടെ കാത്തിരിക്കൂ"
                    },
                    "clips": ["WAIT", "HERE"]
                },
                {
                    "id": "reply_come_tomorrow",
                    "text": {
                        "en": "Please come tomorrow",
                        "ta": "நாளை வாருங்கள்",
                        "hi": "कृपया कल आइए",
                        "te": "రేపు రండి",
                        "kn": "ನಾಳೆ ಬನ್ನಿ",
                        "ml": "നാളെ വരൂ"
                    },
                    "clips": ["COME", "TOMORROW"]
                }
            ]
        },
        "hospital": {
            "name": "Hospital Registration & Helpdesk",
            "description": "Signs and clerk replies for hospital counters, OPD, and pharmacy",
            "signs": [
                "0", "1", "2", "3", "4", "5",
                "HELLO", "THANK YOU", "YES", "NO", "HELP",
                "WHERE", "WHEN", "WATER", "TOILET", "MONEY", "DOCTOR",
                "PAIN", "MEDICINE", "FEVER", "HEAD", "STOMACH"
            ],
            "quick_replies": [
                {
                    "id": "reply_doctor_coming",
                    "text": {
                        "en": "Doctor is coming",
                        "ta": "மருத்துவர் வருகிறார்",
                        "hi": "डॉक्टर आ रहे हैं",
                        "te": "డాక్టర్ వస్తున్నారు",
                        "kn": "ವೈದ್ಯರು ಬರುತ್ತಿದ್ದಾರೆ",
                        "ml": "ഡോക്ടർ വരുന്നുണ്ട്"
                    },
                    "clips": ["DOCTOR", "COME"]
                },
                {
                    "id": "reply_room_3",
                    "text": {
                        "en": "Go to room 3",
                        "ta": "அறை 3க்கு செல்லுங்கள்",
                        "hi": "कमरा नंबर 3 में जाएं",
                        "te": "రూమ్ 3కి వెళ్ళండి",
                        "kn": "ರೂಮ್ 3ಕ್ಕೆ ಹೋಗಿ",
                        "ml": "റൂം 3ലേക്ക് പോകുക"
                    },
                    "clips": ["GO", "ROOM", "3"]
                },
                {
                    "id": "reply_medicine_twice",
                    "text": {
                        "en": "Medicine twice a day",
                        "ta": "மருந்தை ஒரு நாளைக்கு இருமுறை உட்கொள்ளவும்",
                        "hi": "दवा दिन में दो बार लें",
                        "te": "మందు రోజుకు రెండుసార్లు వేసుకోండి",
                        "kn": "ದಿನಕ್ಕೆ ಎರಡು ಬಾರಿ ಔಷಧಿ ತೆಗೆದುಕೊಳ್ಳಿ",
                        "ml": "മരുന്ന് ദിവസം രണ്ടുതവണ കഴിക്കുക"
                    },
                    "clips": ["MEDICINE", "2", "DAY"]
                },
                {
                    "id": "reply_wait_here",
                    "text": {
                        "en": "Please wait here",
                        "ta": "இங்கே காத்திருங்கள்",
                        "hi": "कृपया यहाँ प्रतीक्षा करें",
                        "te": "దయచేసి ఇక్కడ వేచి ఉండండి",
                        "kn": "ದಯವಿಟ್ಟು ಇಲ್ಲಿ ನಿರೀಕ್ಷಿಸಿ",
                        "ml": "ദയവായി ഇവിടെ കാത്തിരിക്കൂ"
                    },
                    "clips": ["WAIT", "HERE"]
                }
            ]
        }
    }
    path = os.path.join(PACKS_DIR, "packs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(packs, f, indent=2, ensure_ascii=False)
    print(f"Generated {path}")


def build_templates():
    templates = {
        "TRAIN WHEN": {
            "en": "When does my train leave?",
            "ta": "என் ரயில் எப்போது புறப்படும்?",
            "hi": "मेरी ट्रेन कब छूटेगी?",
            "te": "నా రైలు ఎప్పుడు బయలుదేరుతుంది?",
            "kn": "ನನ್ನ ರೈಲು ಯಾವಾಗ ಹೊರಡುತ್ತದೆ?",
            "ml": "എന്റെ ട്രെയിൻ എപ്പോഴാണ് പുറപ്പെടുന്നത്?"
        },
        "HELLO TICKET": {
            "en": "Hello, I need to buy a train ticket.",
            "ta": "வணக்கம், எனக்கு ஒரு ரயில் டிக்கெட் வேண்டும்.",
            "hi": "नमस्ते, मुझे एक ट्रेन टिकट चाहिए।",
            "te": "నమస్కారం, నాకు ఒక రైలు టికెట్ కావాలి.",
            "kn": "ನಮಸ್ಕಾರ, ನನಗೆ ಒಂದು ರೈಲು ಟಿಕೆಟ್ ಬೇಕು.",
            "ml": "നമസ്കാരം, എനിക്ക് ഒരു ട്രെയിൻ ടിക്കറ്റ് വേണം."
        },
        "TICKET": {
            "en": "I want to book a ticket.",
            "ta": "எனக்கு ஒரு டிக்கெட் வேண்டும்.",
            "hi": "मुझे टिकट बुक करना है।",
            "te": "నాకు టికెట్ కావాలి.",
            "kn": "ನನಗೆ ಟಿಕೆಟ್ ಬೇಕು.",
            "ml": "എനിക്ക് ഒരു ടിക്കറ്റ് വേണം."
        },
        "PLATFORM WHERE": {
            "en": "Where is the platform located?",
            "ta": "பிளாட்பார்ம் எங்குள்ளது?",
            "hi": "प्लेटफॉर्म कहाँ है?",
            "te": "ప్లాట్‌ఫారమ్ ఎక్కడ ఉంది?",
            "kn": "ಪ್ಲಾಟ್‌ಫಾರ್ಮ್ ಎಲ್ಲಿದೆ?",
            "ml": "പ്ലാറ്റ്‌ഫോം എവിടെയാണ്?"
        },
        "HELP": {
            "en": "Please help me with counter assistance.",
            "ta": "தயவுசெய்து எனக்கு உதவி செய்யுங்கள்.",
            "hi": "कृपया मेरी सहायता करें।",
            "te": "దయచేసి నాకు సహాయం చేయండి.",
            "kn": "ದಯವಿಟ್ಟು ನನಗೆ ಸಹಾಯ ಮಾಡಿ.",
            "ml": "ദയവായി എന്നെ സഹായിക്കൂ."
        },
        "WATER WHERE": {
            "en": "Where can I find drinking water?",
            "ta": "குடிநீர் எங்கே கிடைக்கும்?",
            "hi": "पीने का पानी कहाँ मिलेगा?",
            "te": "తాగునీరు ఎక్కడ దొరుకుతుంది?",
            "kn": "ಕುಡಿಯುವ ನೀರು ಎಲ್ಲಿ ಸಿಗುತ್ತದೆ?",
            "ml": "കുടിവെള്ളം എവിടെ ലഭിക്കും?"
        },
        "WATER": {
            "en": "I need drinking water, please.",
            "ta": "எனக்கு குடிநீர் வேண்டும்.",
            "hi": "मुझे पीने का पानी चाहिए।",
            "te": "నాకు తాగునీరు కావాలి.",
            "kn": "ನನಗೆ ಕುಡಿಯುವ ನೀರು ಬೇಕು.",
            "ml": "എനിക്ക് കുടിവെള്ളം വേണം."
        },
        "TOILET WHERE": {
            "en": "Where is the restroom / toilet?",
            "ta": "கழிப்பறை எங்குள்ளது?",
            "hi": "शौचालय कहाँ है?",
            "te": "మరుగుదొడ్డి ఎక్కడ ఉంది?",
            "kn": "ಶೌಚಾಲಯ ಎಲ್ಲಿದೆ?",
            "ml": "ടോയ്‌ലറ്റ് എവിടെയാണ്?"
        },
        "MONEY": {
            "en": "How much cash / money is required?",
            "ta": "எவ்வளவு பணம் செலுத்த வேண்டும்?",
            "hi": "कितने पैसे देने होंगे?",
            "te": "ఎంత డబ్బు ఇవ్వాలి?",
            "kn": "ಎಷ್ಟು ಹಣ ನೀಡಬೇಕು?",
            "ml": "എത്ര പണം നൽകണം?"
        },
        "DOCTOR HELP": {
            "en": "I need emergency doctor assistance.",
            "ta": "எனக்கு அவசர மருத்துவர் உதவி தேவை.",
            "hi": "मुझे आपातकालीन डॉक्टर की सहायता चाहिए।",
            "te": "నాకు అత్యవసరంగా డాక్టర్ సహాయం కావాలి.",
            "kn": "ನನಗೆ ತಕ್ಷಣ ವೈದ್ಯರ ಸಹಾಯ ಬೇಕು.",
            "ml": "എനിക്ക് അടിയന്തിരമായി ഡോക്ടറുടെ സഹായം വേണം."
        },
        "HEAD PAIN": {
            "en": "I am suffering from a severe headache.",
            "ta": "எனக்கு கடுமையான தலைவலி உள்ளது.",
            "hi": "मुझे बहुत तेज सिरदर्द हो रहा है।",
            "te": "నాకు తీవ్రమైన తలనొప్పిగా ఉంది.",
            "kn": "ನನಗೆ ತೀವ್ರ ತಲೆನೋವು ಇದೆ.",
            "ml": "എനിക്ക് കഠിനമായ തലവേദനയുണ്ട്."
        },
        "STOMACH PAIN": {
            "en": "I have severe stomach pain.",
            "ta": "எனக்கு கடுமையான வயிற்று வலி உள்ளது.",
            "hi": "मुझे पेट में बहुत दर्द है।",
            "te": "నాకు తీవ్రమైన కడుపు నొప్పి ఉంది.",
            "kn": "ನನಗೆ ಹೊಟ್ಟೆ ನೋವು ಇದೆ.",
            "ml": "എനിക്ക് കഠിനമായ വയറുവേദനയുണ്ട്."
        },
        "FEVER MEDICINE": {
            "en": "I have fever and need medicine.",
            "ta": "எனக்கு காய்ச்சல் உள்ளது, மருந்து வேண்டும்.",
            "hi": "मुझे बुखार है और दवा चाहिए।",
            "te": "నాకు జ్వరం ఉంది, మందు కావాలి.",
            "kn": "ನನಗೆ ಜ್ವರವಿದೆ, ಔಷಧಿ ಬೇಕು.",
            "ml": "എനിക്ക് പനിയുണ്ട്, മരുന്ന് വേണം."
        },
        "THANK YOU": {
            "en": "Thank you very much for your help.",
            "ta": "உங்கள் உதவிக்கு மிக்க நன்றி.",
            "hi": "आपकी सहायता के लिए बहुत धन्यवाद।",
            "te": "మీ సహాయానికి చాలా ధన్యవాదాలు.",
            "kn": "ನಿಮ್ಮ ಸಹಾಯಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು.",
            "ml": "നിങ്ങളുടെ സഹായത്തിന് വളരെ നന്ദി."
        },
        "YES": {
            "en": "Yes, I confirm that.",
            "ta": "ஆம், நான் உறுதி செய்கிறேன்.",
            "hi": "हाँ, मैं पुष्टि करता हूँ।",
            "te": "అవును, నేను నిర్ధారిస్తున్నాను.",
            "kn": "ಹೌದು, ನಾನು ದೃಢೀಕರಿಸುತ್ತೇನೆ.",
            "ml": "അതെ, ഞാൻ സ്ഥിരീകരിക്കുന്നു."
        },
        "NO": {
            "en": "No, that is not correct.",
            "ta": "இல்லை, அது சரியல்ல.",
            "hi": "नहीं, यह सही नहीं है।",
            "te": "లేదు, అది సరైనది కాదు.",
            "kn": "ಇಲ್ಲ, ಅದು ಸರಿಯಲ್ಲ.",
            "ml": "അല്ല, അത് ശരിയല്ല."
        }
    }
    path = os.path.join(PACKS_DIR, "templates.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)
    print(f"Generated {path}")


def build_all_assets():
    os.makedirs(PACKS_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)

    build_languages()
    build_regions()
    build_packs()
    build_templates()
    print("All pack and template assets successfully built!")


if __name__ == "__main__":
    build_all_assets()

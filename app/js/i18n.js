/**
 * Citizen-facing interface strings, per spoken language.
 * Sign names and how-to notes live in assets/packs/lexicon.json (served via
 * /api/vocabulary). Missing strings fall back to English.
 */

const STRINGS = {
  en: {
    guideTitle: "Sign guide",
    guideSub: "Tap a sign to add it to the request, or preview it on the hand.",
    commonRequests: "Common requests",
    allSigns: "All signs",
    taughtSigns: "Taught signs",
    add: "Add",
    show: "Show",
    says: "Says:",
    added: "Added",
    guideNote: "Hand tilt and two-handed execution can vary by region. If a sign is not recognized, tap it here instead.",
  },
  ta: {
    guideTitle: "சைகை வழிகாட்டி",
    guideSub: "கோரிக்கையில் சேர்க்க ஒரு சைகையைத் தட்டவும், அல்லது கையில் முன்னோட்டம் பார்க்கவும்.",
    commonRequests: "பொதுவான கோரிக்கைகள்",
    allSigns: "அனைத்து சைகைகள்",
    taughtSigns: "கற்பித்த சைகைகள்",
    add: "சேர்",
    show: "காட்டு",
    says: "பொருள்:",
    added: "சேர்க்கப்பட்டது",
    guideNote: "கை சாய்வும் இரு கை பயன்பாடும் பகுதிக்குப் பகுதி மாறுபடலாம். சைகை அடையாளம் காணப்படவில்லை என்றால், இங்கே தட்டவும்.",
  },
  hi: {
    guideTitle: "संकेत गाइड",
    guideSub: "अनुरोध में जोड़ने के लिए किसी संकेत पर टैप करें, या उसे हाथ पर देखें।",
    commonRequests: "सामान्य अनुरोध",
    allSigns: "सभी संकेत",
    taughtSigns: "सिखाए गए संकेत",
    add: "जोड़ें",
    show: "दिखाएँ",
    says: "अर्थ:",
    added: "जोड़ा गया",
    guideNote: "हाथ का झुकाव और दोनों हाथों का प्रयोग क्षेत्र के अनुसार बदल सकता है। यदि संकेत पहचाना न जाए, तो यहाँ टैप करें।",
  },
  te: {
    guideTitle: "సంజ్ఞ గైడ్",
    guideSub: "అభ్యర్థనకు జోడించడానికి సంజ్ఞను నొక్కండి, లేదా చేతిపై చూడండి.",
    commonRequests: "సాధారణ అభ్యర్థనలు",
    allSigns: "అన్ని సంజ్ఞలు",
    taughtSigns: "నేర్పిన సంజ్ఞలు",
    add: "జోడించు",
    show: "చూపించు",
    says: "అర్థం:",
    added: "జోడించబడింది",
    guideNote: "చేతి వంపు, రెండు చేతుల వాడకం ప్రాంతాన్ని బట్టి మారవచ్చు. సంజ్ఞ గుర్తించబడకపోతే, ఇక్కడ నొక్కండి.",
  },
  kn: {
    guideTitle: "ಸಂಜ್ಞೆ ಮಾರ್ಗದರ್ಶಿ",
    guideSub: "ವಿನಂತಿಗೆ ಸೇರಿಸಲು ಸಂಜ್ಞೆಯನ್ನು ಒತ್ತಿ, ಅಥವಾ ಕೈಯಲ್ಲಿ ನೋಡಿ.",
    commonRequests: "ಸಾಮಾನ್ಯ ವಿನಂತಿಗಳು",
    allSigns: "ಎಲ್ಲಾ ಸಂಜ್ಞೆಗಳು",
    taughtSigns: "ಕಲಿಸಿದ ಸಂಜ್ಞೆಗಳು",
    add: "ಸೇರಿಸಿ",
    show: "ತೋರಿಸಿ",
    says: "ಅರ್ಥ:",
    added: "ಸೇರಿಸಲಾಗಿದೆ",
    guideNote: "ಕೈಯ ಓರೆ ಮತ್ತು ಎರಡು ಕೈಗಳ ಬಳಕೆ ಪ್ರದೇಶದಿಂದ ಪ್ರದೇಶಕ್ಕೆ ಬದಲಾಗಬಹುದು. ಸಂಜ್ಞೆ ಗುರುತಿಸದಿದ್ದರೆ, ಇಲ್ಲಿ ಒತ್ತಿ.",
  },
  ml: {
    guideTitle: "ആംഗ്യ ഗൈഡ്",
    guideSub: "അഭ്യർത്ഥനയിൽ ചേർക്കാൻ ഒരു ആംഗ്യം ടാപ്പ് ചെയ്യുക, അല്ലെങ്കിൽ കൈയിൽ കാണുക.",
    commonRequests: "സാധാരണ അഭ്യർത്ഥനകൾ",
    allSigns: "എല്ലാ ആംഗ്യങ്ങളും",
    taughtSigns: "പഠിപ്പിച്ച ആംഗ്യങ്ങൾ",
    add: "ചേർക്കുക",
    show: "കാണിക്കുക",
    says: "അർത്ഥം:",
    added: "ചേർത്തു",
    guideNote: "കൈയുടെ ചരിവും രണ്ടു കൈകളുടെ ഉപയോഗവും പ്രദേശമനുസരിച്ച് മാറാം. ആംഗ്യം തിരിച്ചറിഞ്ഞില്ലെങ്കിൽ, ഇവിടെ ടാപ്പ് ചെയ്യുക.",
  },
};

export function t(lang, key) {
  return STRINGS[lang]?.[key] ?? STRINGS.en[key] ?? key;
}

/** Sign name in a language from the lexicon; falls back to the sign label. */
export function signWord(lexicon, sign, lang) {
  const entry = lexicon?.words?.[sign];
  return entry?.[lang] || entry?.en || sign;
}

export function signNote(lexicon, sign, lang) {
  const entry = lexicon?.notes?.[sign];
  return entry?.[lang] || entry?.en || "";
}

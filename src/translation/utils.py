# -*- coding: latin-1 -*-
SHOTS = [
    {
        "en": "The quick brown fox jumps over the lazy dog.",
        "de": "Der schnelle braune Fuchs springt ¨¹ber den faulen Hund.",
        "mix": "The schnelle brown fox jumps ¨¹ber the lazy Hund."
    },
    {
        "en": "She sells sea shells by the sea shore.",
        "de": "Sie verkauft Muscheln am Meeresufer.",
        "mix": "She verkauft sea shells am Meeresufer."
    },
    {
        "en": "Knowledge is power, and with great power comes great responsibility.",
        "de": "Wissen ist Macht, und mit gro?er Macht kommt gro?e Verantwortung.",
        "mix": "Wissen ist power, und mit gro?er Macht comes great responsibility."
    },
    {
        "en": "Artificial intelligence is transforming the way we live and work.",
        "de": "K¨¹nstliche Intelligenz ver?ndert die Art, wie wir leben und arbeiten.",
        "mix": "Artificial Intelligenz is transforming the Weg we live und work."
    },
    {
        "en": "The weather today is sunny with a chance of rain in the afternoon.",
        "de": "Das Wetter heute ist sonnig mit einer Regenwahrscheinlichkeit am Nachmittag.",
        "mix": "The Wetter today ist sunny mit a chance of Regen in the afternoon."
    },
    {
        "en": "Learning multiple languages can broaden your horizon and improve cognitive skills.",
        "de": "Mehrere Sprachen zu lernen kann Ihren Horizont erweitern und die kognitive F?higkeit verbessern.",
        "mix": "Learning mehrere languages kann broaden your Horizont und improve cognitive skills."
    },
    {
        "en": "The cat is sitting on the mat and watching the birds outside.",
        "de": "Die Katze sitzt auf der Matte und beobachtet die V?gel drau?en.",
        "mix": "The Katze is sitting on der Matte und watching the V?gel drau?en."
    },
    {
        "en": "Hard work and persistence lead to success in life.",
        "de": "Harte Arbeit und Ausdauer f¨¹hren zu Erfolg im Leben.",
        "mix": "Hard Arbeit und Ausdauer lead zu success im Leben."
    },
    {
        "en": "Our new project will launch next week with a big event.",
        "de": "Unser neues Projekt wird n?chste Woche mit einem gro?en Event gestartet.",
        "mix": "Our neues Projekt will launch n?chste Woche mit einem big Event."
    },
    {
        "en": "The child was playing in the park with their friends all afternoon.",
        "de": "Das Kind spielte den ganzen Nachmittag mit seinen Freunden im Park.",
        "mix": "The Kind was playing im Park mit their Freunden all afternoon."
    },
    {
        "en": "Technology has made global communication faster and more accessible.",
        "de": "Technologie hat die globale Kommunikation schneller und zug?nglicher gemacht.",
        "mix": "Technology hat made globale Kommunikation faster und more zug?nglich."
    },
    {
        "en": "The book you recommended turned out to be very interesting.",
        "de": "Das Buch, das du empfohlen hast, war sehr interessant.",
        "mix": "The Buch you recommended turned out to be sehr interessant."
    },
    {
        "en": "The restaurant offers a wide variety of dishes from different cuisines.",
        "de": "Das Restaurant bietet eine gro?e Auswahl an Gerichten aus verschiedenen K¨¹chen.",
        "mix": "The Restaurant bietet eine wide variety of Gerichten aus different K¨¹chen."
    },
    {
        "en": "The teacher explained the problem with a clear and simple example.",
        "de": "Der Lehrer erkl?rte das Problem mit einem klaren und einfachen Beispiel.",
        "mix": "The Lehrer explained the Problem mit einem clear und simple Beispiel."
    },
    {
        "en": "Traveling allows you to experience new cultures and meet different people.",
        "de": "Reisen erm?glicht es dir, neue Kulturen zu erleben und verschiedene Menschen kennenzulernen.",
        "mix": "Traveling erm?glicht es dir, neue Cultures zu erleben und meet verschiedene Menschen."
    },
    {
        "en": "The movie was so engaging that we didn¡¯t notice the time passing.",
        "de": "Der Film war so fesselnd, dass wir die Zeit nicht vergingen bemerkten.",
        "mix": "The Film was so fesselnd that we didn¡¯t notice die Zeit passing."
    }
]

INSTR_MAP = {
    ("en", "de"): "Translate the English into German. You should directly give the translated sentence. Do not add any greeting words.\n\n{}\n\nInput: {}\nOutput: ",
    ("de", "en"): "Und ?bersetzen sie Deutsch in englisch.. Du sollst einfach den satz ?bersetzen. F?ge keinen gru? hinzu.\n\n{}\n\nImportieren: {}\nExportieren: ",
    ("mix", "en"): "Here is a sentence mixed of English and German. Translate the sentence into English. You should directly give the translated sentence. Do not add any greeting words.\n\n{}\n\n",
    ("mix", "de"): "Here is a sentence mixed of English and German. Translate the sentence into German. You should directly give the translated sentence. Do not add any greeting words.\n\n{}\n\n",
}

LLM_JUDGE_INSTURCTION = """Evaluate the semantic similarity between the following two sentences and provide a score between 1 and 10. Use the following criteria:

- **1**: The sentences are completely unrelated, with no semantic connection.
- **2-3**: The sentences are almost unrelated, with very little semantic similarity.
- **4-5**: The sentences share some similarity, but their overall meaning is quite different.
- **6-7**: The sentences are semantically similar, though there are some notable differences.
- **8-9**: The sentences are very similar, with only slight differences in meaning.
- **10**: The sentences are identical in meaning, with no semantic differences.

Please provide the score wrapped in <Score></Score> directly. Do not add any greeting words and analysis!

# Example:
## Input
Sentence 1: Ich habe heute Morgen Kaffee getrunken.
Sentence 2: I drank coffee this morning.
## Output
<Score>9</Score>

# Task
## Input
Sentence 1: {}
Sentence 2: {}
## Output
"""

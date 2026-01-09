# ai_layer.py
import os
import json
from typing import Optional, Dict

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# =========================
# CONFIG
# =========================

AI_MODEL = "gpt-4.1-mini"
AI_TIMEOUT = 8
AI_CONFIDENCE_DEFAULT = 0.5


# =========================
# CLIENT
# =========================

_client = None

def _get_client():
    global _client

    if _client is not None:
        return _client

    if OpenAI is None:
        print("⚠️ OpenAI SDK non installato")
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY mancante")
        return None

    try:
        _client = OpenAI(api_key=api_key)
        return _client
    except Exception as e:
        print("❌ ERRORE INIT OpenAI:", e)
        return None


# =========================
# PROMPT
# =========================

SYSTEM_PROMPT = """
Sei un assistente che interpreta messaggi di clienti di una pizzeria
per aiutare un sistema di prenotazioni automatico.

Il tuo compito NON è gestire la conversazione,
ma ESTRARRE informazioni strutturate dal testo dell’utente.

REGOLE OBBLIGATORIE:

1) NON calcolare MAI date reali.
2) Se l’utente usa date relative o testuali (oggi, domani, sabato, ecc.),
   restituisci ESATTAMENTE il testo nel campo "data_testuale".
3) NON inventare dati
4) Se un dato non è esplicitamente presente, usa null.
5) NON inventare informazioni.
6) Restituisci SOLO JSON valido.
"""


USER_PROMPT = """
Messaggio del cliente:
"{text}"

Rispondi SOLO con questo JSON:

{
  "nome": null,
  "cognome": null,
  "persone": null,
  "data_testuale": null,
  "ora": null,
  "confidence": 0.0
}
"""


# =========================
# FUNZIONE PRINCIPALE
# =========================

def interpreta_con_ai(testo: str) -> Optional[Dict]:
    if not testo or not testo.strip():
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        response = client.responses.create(
            model=AI_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": USER_PROMPT.replace("{text}", testo.strip())}
            ],
            timeout=AI_TIMEOUT
        )

        # =========================
        # 🔒 ESTRAZIONE TESTO (SDK SAFE)
        # =========================
        raw = None

        if hasattr(response, "output") and response.output:
            block = response.output[0]
            if hasattr(block, "content") and block.content:
                item = block.content[0]

                # CASO 1: il modello ha già restituito un dict
                if isinstance(item, dict):
                    data = item
                # CASO 2: testo JSON
                elif hasattr(item, "text"):
                    raw = item.text
                else:
                    raw = None
        else:
            raw = None

        # =========================
        # 🔒 SE È GIÀ UN DICT → OK
        # =========================
        if isinstance(raw, dict):
            data = raw
        else:
            if not raw or not isinstance(raw, str):
                print("❌ RISPOSTA AI NON TESTUALE:", response)
                return None

            # =========================
            # 🔒 ESTRAZIONE JSON ROBUSTA
            # =========================
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                json_txt = raw[start:end]
                data = json.loads(json_txt)
            except Exception:
                print("❌ JSON NON VALIDO DALL'AI:", raw)
                return None

        # =========================
        # 🔒 HARDENING FINALE
        # =========================
        if not isinstance(data, dict):
            return None

        data.setdefault("nome", None)
        data.setdefault("cognome", None)
        data.setdefault("persone", None)
        data.setdefault("data_testuale", None)
        data.setdefault("ora", None)
        data.setdefault("confidence", AI_CONFIDENCE_DEFAULT)

        return data

    except Exception as e:
        print("❌ ERRORE AI:", e)
        return None

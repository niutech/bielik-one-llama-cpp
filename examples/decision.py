import requests

payload = {
    "model": "decision",
    "state": "Faktura 12/2026 na 4 300 EUR nie została opłacona. Termin płatności minął wczoraj. Klient spóźniał się z płatnością dwa razy w tym roku.",
    "questions": {
        "status": {
            "type": "choice",
            "instructions": "Jaki jest status faktury?",
            "criteria": {
                "paid": "Opłacona",
                "overdue": "Nieopłacona, termin minął",
                "not_due": "Nieopłacona, termin jeszcze nie minął",
            },
        },
        "deadline_passed": {
            "type": "noul",
            "instructions": "Czy termin płatności minął?",
            "criteria": {"false": "Nie", "true": "Tak"},
        },
        "urgency": {
            "type": "score",
            "instructions": "Jak pilna jest windykacja?",
            "criteria": ["brak", "niska", "średnia", "wysoka", "krytyczna"],
        },
        "actions": {
            "type": "labels",
            "instructions": "Które działania należy teraz podjąć?",
            "criteria": {
                "reminder": "Wysłać przypomnienie o płatności",
                "interest": "Naliczć odsetki za opóźnienie",
                "cancel": "Anulować fakturę",
            },
        },
    },
}

response = requests.post("http://127.0.0.1:8026/v1/systemone", json=payload, timeout=120)
response.raise_for_status()
print(response.json())

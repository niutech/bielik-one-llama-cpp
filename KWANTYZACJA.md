# Bielik-One GGUF: 11B, 4B i kalibracja uwzględniająca kwantyzację

Backend oparty na llama.cpp odczytuje natywne identyfikatory tokenów `<|systemone_decision_XXXX|>` z tokenizera GGUF. Nie może zakładać, że identyfikatory tokenów ani rozmiary słownika są identyczne dla modeli o różnych rozmiarach: checkpoint 11B v3.1-r1 udostępnia 651 tokenów decyzyjnych, a checkpoint 4B r1 — 516. Skanowanie słownika przez backend oraz polecenie `check` weryfikują rzeczywistą zawartość pliku przed uruchomieniem serwera.

## Obsługiwane repozytoria GGUF

- 11B: [`jglowa/Bielik-One-11B-v3.1-r1-GGUF`](https://huggingface.co/jglowa/Bielik-One-11B-v3.1-r1-GGUF)
- 4B: [`jglowa/Bielik-One-4B-r1-GGUF`](https://huggingface.co/jglowa/Bielik-One-4B-r1-GGUF)

W obu repozytoriach dostępne są obecnie pliki `BF16`, `Q8_0` i `Q4_K_M`. Przykładowe polecenia pobierania:

```bash
# 11B — wybierz jeden wariant kwantyzacji (Q4_K_M, Q8_0 lub BF16)
hf download jglowa/Bielik-One-11B-v3.1-r1-GGUF \
  Bielik-One-11B-v3.1-r1-Q4_K_M.gguf --local-dir .

# 4B — wybierz jeden wariant kwantyzacji (Q4_K_M, Q8_0 lub BF16)
hf download jglowa/Bielik-One-4B-r1-GGUF \
  Bielik-One-4B-r1-Q4_K_M.gguf --local-dir .
```

Aby pobrać inne warianty, użyj tych samych poleceń, zmieniając fragment nazwy pliku na `Q8_0` lub `BF16`. Nie jest wymagana żadna dodatkowa, specyficzna dla modelu biblioteka Pythona — używaj tego samego backendu `llama-cpp-python`.

Sprawdź pobrany plik i uruchom serwer:

```bash
bielik-one-llama-cpp check ./Bielik-One-4B-r1-Q4_K_M.gguf
bielik-one-llama-cpp serve ./Bielik-One-4B-r1-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8026 --n-ctx 32768 --n-gpu-layers -1
```

Polecenie `check` wyświetla wykrytą rodzinę modelu i kwantyzację, liczbę natywnych tokenów decyzyjnych oraz zakres identyfikatorów tokenów. Pole `legacy_11b_token_id_range_matches` służy wyłącznie do porównania z układem identyfikatorów starego modelu 11B; dla poprawnego układu 4B oczekiwana wartość to `false`. Jeśli `decision_vocabulary_validated: true` i status jest poprawny, oznacza to, że natywne tokeny w pliku GGUF zostały odnalezione i zweryfikowane. Jeśli backend nie znajdzie ciągłego słownika natywnych tokenów decyzyjnych, przerwie uruchamianie serwera i wyświetli konkretny komunikat o błędzie, zamiast po cichu używać nieprawidłowych identyfikatorów tokenów.

## Zasady automatycznej kalibracji

Skalowanie temperaturą zmienia raportowane prawdopodobieństwa, ale nie zmienia wybranej klasy `argmax`. Temperatury zależą jednak od modelu i profilu serwowania:

- Oficjalne temperatury są automatycznie stosowane wyłącznie dla `11B BF16`.
- Pasujący plik dodatkowy (sidecar) o nazwie `<nazwa-pliku-GGUF-bez-rozszerzenia>.calibration.json` jest automatycznie wczytywany dla dowolnego modelu i wariantu kwantyzacji.
- Jeśli nie ma pasującego pliku dodatkowego dla `Q4_K_M`, `Q8_0`, 4B BF16 lub pliku o nierozpoznanej nazwie, silnik używa surowych prawdopodobieństw (`T=1`) i zapisuje w logu informację o braku pasującej kalibracji.
- Opcja `--no-calibration` zawsze wyłącza skalowanie temperaturą. Opcja `--calibration PATH` jawnie wskazuje plik kalibracji.

To celowo ostrożne ustawienie domyślne: użycie temperatur wyznaczonych dla 11B BF16 w modelu 4B lub modelu skwantyzowanym sugerowałoby istnienie danych potwierdzających kalibrację, których w rzeczywistości nie ma.

### Wyznaczanie kalibracji dla konkretnego pliku GGUF

Przygotuj walidacyjny zbiór danych w formacie JSONL. Każdy wiersz zawiera jedno żądanie inferencji i odpowiadającą mu poprawną odpowiedź (lub poprawne odpowiedzi):

```json
{
  "state": "Faktura jest po terminie, a klient nie odpowiedział na przypomnienie.",
  "questions": {
    "status": {
      "type": "choice",
      "instructions": "Jaki jest status faktury?",
      "criteria": {"paid": "Opłacona", "overdue": "Po terminie"}
    },
    "late": {
      "type": "noul",
      "instructions": "Czy termin płatności minął?",
      "criteria": {"true": "Tak", "false": "Nie"}
    },
    "urgency": {
      "type": "score",
      "instructions": "Jak pilna jest sprawa?",
      "criteria": ["niska", "średnia", "wysoka"]
    },
    "actions": {
      "type": "labels",
      "instructions": "Które działania są właściwe?",
      "criteria": {"reminder": "Wyślij przypomnienie", "interest": "Nalicz odsetki"}
    }
  },
  "expected": {
    "status": "overdue",
    "late": true,
    "urgency": 2,
    "actions": ["reminder"]
  }
}
```

Dozwolone wartości referencyjne:

- `choice`: klucz wybranej opcji (na przykład `"overdue"`).
- `noul`: wartość logiczna JSON — `true` lub `false`.
- `score`: indeks całkowity liczony od zera, wskazujący pozycję w `criteria`.
- `labels`: lista kluczy wybranych opcji albo mapa zawierająca wszystkie klucze opcji i przypisane im wartości logiczne.

Domyślnie używaj co najmniej 20 oznaczonych przykładów dla każdego typu pytania. Mechanizm dopasowujący korzysta z predykcji surowych (`T=1`), optymalizuje jedną temperaturę dla każdego typu za pomocą ograniczonego, jednowymiarowego minimalizatora NLL, a następnie zapisuje plik dodatkowy właściwy dla danego modelu obok pliku GGUF:

```bash
bielik-one-llama-cpp calibrate ./Bielik-One-4B-r1-Q4_K_M.gguf \
  --dataset ./validation.jsonl \
  --min-samples-per-type 20
```

Domyślnie wynik zostanie zapisany w pliku `./Bielik-One-4B-r1-Q4_K_M.calibration.json`. Uruchom ponownie API z automatyczną kalibracją:

```bash
bielik-one-llama-cpp serve ./Bielik-One-4B-r1-Q4_K_M.gguf \
  --calibration auto --host 127.0.0.1 --port 8026
```

Wszystkie wiersze muszą używać `permutations=1` oraz tych samych ustawień `read_twice` i `questions_before_state`. Mechanizm dopasowujący odrzuca inne profile permutacji, zamiast tworzyć temperaturę niedopasowaną do sposobu późniejszego serwowania. Dla `labels.mode="per_option"` dopasowywana jest temperatura binarna, ponieważ ten tryb korzysta z głowicy binarnej; domyślna głowica `labels` otrzymuje własną temperaturę sigmoidalną. Dopasowuj temperatury wyłącznie na wydzielonej części walidacyjnej i zachowaj osobny zbiór testowy do końcowej oceny. Jeśli liczba przykładów danego typu jest mniejsza niż skonfigurowane minimum, mechanizm pozostawia dla tego typu `T=1`. Nie tworzy fikcyjnych wartości kalibracji, gdy brakuje etykiet.

### Przykład: użycie ręcznie przygotowanej kalibracji

```bash
bielik-one-llama-cpp serve ./Bielik-One-11B-v3.1-r1-Q8_0.gguf \
  --calibration ./calibration-11b-q8.json
```

Aby porównać prawdopodobieństwa surowe i skalibrowane, uruchom drugą instancję z opcją `--no-calibration`. Wraz z wynikami benchmarków zapisuj nazwę pliku GGUF i wariant kwantyzacji, wersję kompilacji `llama-cpp-python`, rozmiar kontekstu oraz opcje promptu.

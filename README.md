# Bielik-One-llama-cpp

Silnik decyzji dla **`speakleash/Bielik-One-11B-v3.1-r1`** z backendem **`llama-cpp-python`** zamiast vLLM. Projekt udostępnia zgodny kształt żądania `POST /v1/systemone` i cztery typy odpowiedzi: `choice`, `noul`, `score`, `labels`.

> **Status implementacji:** repozytorium zawiera kompletną warstwę serwera, walidację, readout logitów, obliczanie rozkładów, temperatury kalibracyjne, CLI i testy jednostkowe bez GPU.

## Co robi silnik

- Wczytuje **scalony model Bielik-One** z pliku GGUF przez `llama-cpp-python`.
- Odczytuje logity tokenów `<|systemone_decision_XXXX|>` bez generowania tekstu. Używa niskopoziomowego `llama_get_logits_ith(ctx, -1)`, więc nie alokuje pełnej macierzy `n_ctx × vocab` w `Llama.scores`.
- Implementuje `choice` (softmax po opcjach), `noul` (prawdopodobieństwo `true`), `score` (rozkład na uporządkowanej skali i wartość oczekiwaną) i `labels` (niezależny sigmoid dla każdej etykiety).
- `labels.mode="head"` ocenia etykiety jednym przebiegiem. `labels.mode="per_option"` wykonuje osobny odczyt binarny dla każdej etykiety.
- Obsługuje opcje żądania `permutations` (1–16), `read_twice` oraz `questions_before_state`.
- Współdzieli rzeczywisty prefiks KV cache llama.cpp pomiędzy pytaniami o ten sam stan. Dla domyślnego układu stan jest przetwarzany raz, a kolejne pytania dokładają tylko swój sufiks.
- Zwraca `probabilities`, `selected` (wybrany klucz) i `confidence` zgodnie z formułami opisanymi w README oryginalnego projektu.
- Domyślnie wystawia API na `127.0.0.1:8026`, z `/health`, `/v1/models` i dokumentacją OpenAPI pod `/docs`.

## Instalacja

### 1. Zainstaluj llama-cpp-python

Projekt wymaga `llama-cpp-python>=0.3.16`, ponieważ backend pobiera logity ostatniego tokenu przez `llama_get_logits_ith`. Starsze wersje nie są obsługiwane.

Wariant CPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[test]'
```

Dla NVIDIA CUDA zbuduj `llama-cpp-python` z obsługą CUDA (wersję i flagi dobierz do lokalnego toolkitu CUDA):

```bash
CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1 \
  pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
pip install -e '.[test]'
```

Dla macOS/Metal:

```bash
CMAKE_ARGS='-DGGML_METAL=on' FORCE_CMAKE=1 \
  pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
pip install -e '.[test]'
```

### 2. Pobierz wybrany plik GGUF

Wybierz kwantyzację (`BF16`, `Q8_0` albo `Q4_K_M`) pliku GGUF do pobrania z [Hugging Face](https://huggingface.co/jglowa/Bielik-One-11B-v3.1-r1-GGUF).

```bash
pip install -U huggingface_hub
hf download jglowa/Bielik-One-11B-v3.1-r1-GGUF Bielik-One-11B-v3.1-r1-BF16.gguf --local-dir .
```

### 4. Sprawdź tokeny decyzyjne i uruchom API

```bash
bielik-one-llama-cpp check ./Bielik-One-11B-v3.1-r1-BF16.gguf

bielik-one-llama-cpp serve ./Bielik-One-11B-v3.1-r1-BF16.gguf \
  --host 127.0.0.1 --port 8026 --n-ctx 32768 --n-gpu-layers -1
```

`--n-gpu-layers -1` próbuje umieścić wszystkie warstwy na GPU. Dla CPU ustaw `--n-gpu-layers 0`; dla częściowego offloadu podaj odpowiednią liczbę. Można też ograniczyć `--n-ctx`, `--n-batch` i `--n-ubatch`, aby dopasować pamięć do sprzętu.

Następnie sprawdź:

```bash
curl http://127.0.0.1:8026/health
```

## API

### Przykładowe żądanie

```bash
curl -s http://127.0.0.1:8026/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "decision",
    "state": "Faktura 12/2026 na 4300 EUR nie została opłacona. Termin minął wczoraj.",
    "questions": {
      "status": {
        "type": "choice",
        "instructions": "Jaki jest status faktury?",
        "criteria": {
          "paid": "Opłacona",
          "overdue": "Nieopłacona, termin płatności minął",
          "not_due": "Nieopłacona, termin jeszcze nie minął"
        }
      },
      "late": {
        "type": "noul",
        "instructions": "Czy termin płatności minął?",
        "criteria": {"false": "Nie", "true": "Tak"}
      },
      "urgency": {
        "type": "score",
        "instructions": "Jak pilna jest windykacja?",
        "criteria": ["brak", "niska", "średnia", "wysoka", "krytyczna"]
      },
      "actions": {
        "type": "labels",
        "instructions": "Które działania są teraz właściwe?",
        "criteria": {
          "reminder": "Wysłać przypomnienie",
          "interest": "Nalicz odsetki",
          "cancel": "Anulować fakturę"
        }
      }
    }
  }'
```

### Kształt odpowiedzi

Odpowiedź zawiera mapę `answers`, indeksowaną nazwą pytania. Pola wynikowe:

- `choice`: wybrany klucz `choice`, rozkład `probabilities`, `confidence`.
- `noul`: prawdopodobieństwo prawdy w polu `noul`; `probabilities` zawiera `false` i `true`. Logity mają kanoniczną kolejność `true`, `false`, niezależnie od kolejności kluczy w JSON; bez `criteria` silnik używa opisów `Yes` i `No`.
- `score`: wartość oczekiwana poziomu (indeksy od 0), rozkład `probabilities`, mapa `legend`, `confidence`.
- `labels`: niezależne prawdopodobieństwa, lista `selected` (domyślny próg 0.5), `confidence` jako prawdopodobieństwo poprawności całego zestawu, `mode`.

`confidence` opisuje koncentrację rozkładu, a nie jest niezależną gwarancją prawdopodobieństwa poprawności. Dla decyzji wpływających na ludzi lub finanse stosuj walidację domenową i zatwierdzenie przez człowieka.

### Opcje sposobu odczytu (zgodność z upstream)

Opcje są ustawiane na poziomie całego żądania `POST /v1/systemone`:

```json
{
  "state": "Treść wspólnego stanu...",
  "permutations": 4,
  "read_twice": true,
  "questions_before_state": false,
  "questions": {
    "decision": {
      "type": "choice",
      "instructions": "Która kategoria pasuje?",
      "criteria": {"a": "A", "b": "B", "c": "C"}
    }
  }
}
```

- `permutations` (domyślnie `1`, zakres `1..16`) wykonuje odczyty z różnym ułożeniem opcji. Wyniki każdego odczytu są mapowane do pierwotnych kluczy, kalibrowane i uśredniane jako prawdopodobieństwa. Pierwszy przebieg zachowuje kolejność wejściową; pozostałe kolejności są deterministyczne dla nazwy pytania, typu i kluczy opcji, aby powtarzane żądania dawały powtarzalny układ.
- Dla opcji binarnych istnieją tylko dwa różne porządki. `labels.mode="per_option"` używa najwyżej tych dwóch porządków na kandydata, nawet gdy `permutations` ma większą wartość.
- `read_twice: true` umieszcza stan w promptcie dwukrotnie przed częścią decyzyjną (albo — przy `questions_before_state: true` — po niej). To jeden forward z dłuższym promptem, a nie dwa niezależne odczyty.
- `questions_before_state: true` przenosi całe pytanie wraz z opcjami przed stan. W tym układzie nie można współdzielić prefiksu zawierającego stan, ponieważ stan występuje po zmiennej treści pytania; cache nadal może zachować wspólny nagłówek.
- Przy domyślnym `questions_before_state: false` wspólny prefiks kończy się po bloku stanu (także po jego powtórzeniu, jeżeli `read_twice` jest aktywne). Backend utrzymuje KV cache llama.cpp, usuwa poprzedni sufiks na granicy prefiksu i dekoduje tylko nowy sufiks pytania. Jeżeli dany build llama.cpp nie pozwala na częściowe usunięcie KV, backend bezpiecznie przelicza prompt od początku.

Ograniczenia zgodne z upstream: maksymalnie 64 pytania, 16 permutacji, 1024 odczyty na żądanie oraz 2 000 000 znaków treści powtarzanej przez `labels.mode="per_option"`. Budżet odczytów uwzględnia wszystkie pytania, permutacje i odczyty per-opcja. Powtórzenie stanu nie zwiększa liczby odczytów, ale zwiększa długość promptu i zużycie kontekstu.

Uwaga: zgodność semantyki opcji nie oznacza identycznego promptu ani identycznych wyników liczbowych. Implementacja permutacji jest deterministyczna, ale nie potwierdziłem tożsamości kolejności pseudolosowej z wewnętrznym algorytmem upstream. Ponadto prompt zadaniowy nadal wymaga porównania na rzeczywistym GGUF z upstreamowym silnikiem.

### Kalibracja

W katalogu znajduje się `calibration.json` z temperaturami z oficjalnego pliku modelu: `choice=2.1067`, `binary=1.8987`, `score=2.7321`, `labels=1.0`. Są one dopasowane do referencyjnego serwowania BF16 i oryginalnego promptu `bielik-one`. Nie są gwarantowane dla Q4/Q8 ani dla promptu z tej implementacji. Dla kwantyzowanego GGUF zacznij od surowych wyników:

```bash
bielik-one-llama-cpp serve Bielik-One-11B-v3.1-r1-Q4_K_M.gguf --no-calibration
```

Po zebraniu lokalnego zbioru walidacyjnego dopasuj osobne temperatury dla `choice`, `binary`, `score` i `labels`, a następnie podaj plik przez `--calibration calibration-local.json`. Nie należy traktować samych temperatur BF16 jako kalibracji dla kwantyzacji.

## Parametry

```text
bielik-one-llama-cpp serve Bielik-One-11B-v3.1-r1-BF16.gguf
  --host 127.0.0.1
  --port 8026
  --n-ctx 32768
  --n-gpu-layers -1
  --n-threads N
  --n-batch 2048
  --n-ubatch 512
  --calibration PATH
  --no-calibration
```

Ograniczenia wejścia: maks. 64 pytania, 651 opcji na typ, 400 000 znaków stanu, 400 000 znaków pytań oraz 1024 odczyty decyzyjne na żądanie. Treści użytkownika są traktowane jako dane i neutralizowane z markerów ChatML przed umieszczeniem w promptcie.

## Testy

Testy jednostkowe nie wymagają modelu ani GPU:

```bash
pip install -e '.[test]'
pytest -q
```

Testy porównawcze jakości wymagają wczytanego GGUF i oznaczonego zbioru walidacyjnego. Rekomendowane porównanie: dokładność decyzji, NLL/Brier/ECE, zgodność z upstream `bielik-one` oraz odsetek decyzji przy progach operacyjnych.

## Różnice względem `bielik-one`

- Backend vLLM i plugin vLLM zastąpiono bezpośrednim odczytem logitów z `llama-cpp-python`.
- Obsługiwany jest wskazany **merged model**. Model `-Adapter` wymaga oddzielnej obsługi LoRA i nie jest celem tej wersji.
- API koncentruje się na `/v1/systemone`; nie implementuje proxy zwykłego czatu `/v1/chat/completions` ani dwóch procesów `backend`/`frontend`.
- Prompt zadaniowy tej wersji jest prostym promptem ChatML i wymaga walidacji względem oryginalnego promptu publikowanego w upstreamie przed oczekiwaniem identycznych prawdopodobieństw.
- Request-level features `permutations`, `read_twice` i `questions_before_state` są zaimplementowane. W odróżnieniu od prostego cache tekstowego backend ponownie wykorzystuje cache KV llama.cpp i cofa go do wspólnego prefiksu tokenów.
- Generator permutacji jest deterministyczny i uwzględnia tylko unikalne kolejności (dla dwóch opcji najwyżej dwie). Wewnętrzny algorytm doboru kolejności w upstream nie został potwierdzony na poziomie źródła, dlatego identyczność pojedynczych permutacji nie jest deklarowana.

## Źródła modelu i protokołu

- Model: https://huggingface.co/speakleash/Bielik-One-11B-v3.1-r1
- Oryginalny engine: https://github.com/speakleash/bielik-one
- Licencja wag i oryginalnego projektu: Apache-2.0. Licencja wag pozostaje niezależna od licencji tej implementacji.

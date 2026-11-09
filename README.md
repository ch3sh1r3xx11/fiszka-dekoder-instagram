# Fiszka z Instagrama

**Model AI wrzucony na Instagrama w 6 zrzutach ekranu.** Zrób zrzuty, uruchom dekoder, masz działający model.

Fiszka to mały klasyfikator tematów tekstów o AI z laboratorium [aipoc lab](https://aipoclab.creativeaidrivenpm.com/fiszka):
dostaje wektor tekstu i mówi, o czym tekst jest (17 tematów: agenci, modele językowe, ewaluacja, robotyka…).
Ma **252 194 parametry**. Wszystkie leżą na Instagramie w sześciu postach, a siatka profilu układa się w tę liczbę:

```
2 5 2
1 9 4
```

## Jak to jest zapisane

- Każda waga po **5 bitach** (32 poziomy na warstwę), razem ok. **158 KB**.
- Kafel 1080×1350 niesie **1/6 modelu** (ok. 42 tys. wag): kropki 3×3 px w **4 jasnościach**, czyli 2 bity na kropkę.
- Cyfra jest namalowana **barwą** (błękit / fuksja), dane siedzą w **jasności**. JPEG Instagrama zostawia duże plamy
  barwy i prawie całą jasność.
- Na początku każdego kafla nagłówek (numer kafla, skale warstw), zapisany 3 razy.
- Znaczniki w rogach pozwalają znaleźć kafel na zwykłym zrzucie ekranu.

## Uruchomienie

```bash
pip install -r requirements.txt
python dekoder.py zrzut1.png zrzut2.png zrzut3.png zrzut4.png zrzut5.png zrzut6.png
```

Zrzuty: cały ekran telefonu, post na pełną szerokość (zwykły widok posta). Kolejność dowolna.
Wynik: `fiszka_z_instagrama.npz` z wagami `W1, b1, W2, b2, W3, b3, W4, b4, S, s`.

Otagowanie tekstu (potrzebna lokalna [Ollama](https://ollama.com) z EmbeddingGemma 2, 270M):

```bash
python dekoder.py zrzut*.png --tekst "Tytuł artykułu | treść artykułu"
```

Sam model to 15 linii (`fiszka()` w `dekoder.py`): trzy warstwy ReLU 768→256→128→64, wyjście na 17 tematów
plus połączenie skrótowe z wejścia, sigmoida. Temat, gdy pewność ≥ 0,5, najwyżej 3.

## Czy to działa?

Prawie. To dowód koncepcji, bez korekcji błędów.

| Test (9.11.2026) | Źle odczytane wagi |
|---|---|
| zwykły zrzut feedu z iPhone'a (828 px szerokości) | 2,8% |
| symulacja: JPEG jak Instagram + zrzut 1179 px | ok. 0,5% |

Model złożony ze zrzutów rozpoznał tematy prawie jak oryginał: na tekstach testowych te same tematy albo jeden
dodatkowy. Większość różnicy robi sama kwantyzacja do 5 bitów, nie błędy odczytu.

## Licencja

Kod: MIT. Pełny model, karta modelu i opis treningu: Hugging Face i GitHub labu (od 14.11.2026).

"""Fiszka z Instagrama: odtwarza model AI ze zrzutów ekranu.

Fiszka to mały klasyfikator tematów tekstów o AI z laboratorium aipoc lab (252 194 parametry). Cały model
leży na Instagramie w 6 kaflach: wagi po 5 bitów, zapisane w kropkach 3x3 px w 4 jasnościach (2 bity na
kropkę). Cyfra na kaflu (4, 9, 1, 2, 5, 2 -> siatka profilu 252 194) jest namalowana barwą, dane siedzą w
jasności kropek.

Ten plik: zwykłe zrzuty ekranu (całego ekranu telefonu, post na pełną szerokość) -> fiszka_z_instagrama.npz.
Kafle mogą być w dowolnej kolejności; brakujący kafel = wagi zero (model działa gorzej, ale działa).
Korekcji błędów nie ma — to dowód koncepcji: na prawdziwym zrzucie z iPhone'a ok. 3% wag wychodzi źle.

    python dekoder.py zrzut1.png zrzut2.png ... zrzut6.png
    python dekoder.py zrzut*.png --tekst "Tytuł | treść tekstu o AI"     # z Ollamą i embeddinggemma

Licencja kodu: MIT. Projekt: https://aipoclab.creativeaidrivenpm.com/fiszka
"""
import argparse
import json
import math
import struct
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── układ kafla (taki sam jak w koderze) ──────────────────────────────────────────────────────────
SZER, WYS, K = 1080, 1350, 3
X0, Y0, X1, Y1 = 24, 150, 1056, 1326
COLS, ROWS = (X1 - X0) // K, (Y1 - Y0) // K
CYFRY = [4, 9, 1, 2, 5, 2]                    # cyfra kafla nr 1..6
KSZTALTY = [(768, 256), (256,), (256, 128), (128,), (128, 64), (64,), (64, 17), (17,), (768, 17), (17,)]
NAZWY = ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "S", "s"]
TEMATY = ["llm", "ml", "evaluation", "agents", "multimodal", "security", "reinforcement-learning", "rag", "robotics",
          "hardware", "tooling", "healthcare", "business", "automation", "policy", "open-source", "education"]
MAGIA = b"FSZK"
NAGLOWEK = struct.Struct("<4sBBBIII10f")      # magia, wersja, nr, ile kafli, wag razem, start, ile w kaflu, 10 skal
KOPIE_NAGLOWKA = 3
BITY = 5
POZIOMY = 2 ** BITY
Y_TLO = [18, 52, 86, 120]
Y_CYF = [105, 134, 162, 190]
TLO_BARWA = (30, 60, 110)

# ── cyfry „bambus”: kreski na siatce 6 x 10, ścięte końce, przerwy jak kolanka bambusa ──────────────
GLIFY = {
    1: [((1.3, 2.7), (3.3, 0.7)), ((3.9, 0.6), (3.9, 9.5)), ((2.0, 9.8), (5.6, 9.8))],
    2: [((0.8, 0.8), (4.6, 0.8)), ((5.3, 1.4), (5.3, 4.4)), ((4.9, 5.1), (1.1, 8.9)), ((0.7, 9.6), (5.7, 9.6))],
    4: [((4.2, 0.6), (0.6, 6.0)), ((0.6, 6.7), (5.8, 6.7)), ((4.5, 1.0), (4.5, 9.7))],
    5: [((1.5, 0.8), (5.6, 0.8)), ((0.9, 1.3), (0.9, 4.5)), ((1.5, 5.0), (4.6, 5.0)), ((5.2, 5.6), (5.2, 8.6)),
        ((4.6, 9.4), (0.6, 9.4))],
    9: [((1.4, 0.8), (4.8, 0.8)), ((0.8, 1.4), (0.8, 4.4)), ((1.4, 5.1), (4.8, 5.1)), ((5.4, 1.3), (5.4, 8.5)),
        ((5.0, 9.1), (2.6, 9.9))],
}
GRUBOSC, KOLANKO, PRZERWA = 1.35, 2.3, 0.36


def rysuj_glif(cyfra, d, x0, y0, u):
    for (ax, ay), (bx, by) in GLIFY[cyfra]:
        dx, dy = bx - ax, by - ay
        dl = math.hypot(dx, dy)
        ux, uy = dx / dl, dy / dl
        nx, ny = -uy, ux
        h = GRUBOSC / 2
        sk = h * 0.75
        n_kawal = max(1, round((dl + PRZERWA) / (KOLANKO + PRZERWA)))
        kawal = (dl - PRZERWA * (n_kawal - 1)) / n_kawal
        for k in range(n_kawal):
            t0 = k * (kawal + PRZERWA)
            t1 = t0 + kawal
            pkt = []
            for t, znak in ((t0, 1), (t1, 1), (t1, -1), (t0, -1)):
                tt = min(max(t + sk * znak, 0 if t == t0 else t0 + 0.01), dl)
                pkt.append((x0 + (ax + ux * tt + nx * h * znak) * u, y0 + (ay + uy * tt + ny * h * znak) * u))
            d.polygon(pkt, fill=255)


def maska_cyfry(nr):
    """Które kropki leżą w cyfrze (tam paleta jest jaśniejsza)."""
    m = Image.new("L", (SZER, WYS), 0)
    u = ((Y1 - Y0) * 0.84) / 10.4
    rysuj_glif(CYFRY[nr - 1], ImageDraw.Draw(m), (SZER - 6.4 * u) / 2, Y0 + ((Y1 - Y0) - 10.4 * u) / 2, u)
    m = np.array(m.filter(ImageFilter.MaxFilter(5)))
    return m[Y0 + 1:Y0 + ROWS * K:K, X0 + 1:X0 + COLS * K:K][:ROWS, :COLS] > 128


def jasnosc(c):
    return float(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])


def kolor_o_jasnosci(barwa, y):
    b = np.array(barwa, dtype=np.float32)
    c = b * (y / jasnosc(b))
    if c.max() > 255:
        c = c / c.max() * 255
        c = c + (y - jasnosc(c))
    return np.clip(c, 0, 255)


def jasnosci_palety(nr):
    barwa = (232, 26, 122) if nr % 2 else (67, 188, 249)      # fuksja / błękit
    return ([jasnosc(kolor_o_jasnosci(barwa, y)) for y in Y_CYF],
            [jasnosc(kolor_o_jasnosci(TLO_BARWA, y)) for y in Y_TLO])


# ── odczyt ────────────────────────────────────────────────────────────────────────────────────────
def wyrownaj(obraz):
    """Zrzut -> 1080 x 1350. Szerokość: post na całą szerokość ekranu. Pion osobno, od znacznika do końca
    kropek (na iPhonie skala pionowa różniła się od poziomej o ok. 1,4%)."""
    im = obraz.convert("RGB")
    if im.width != SZER:
        im = im.resize((SZER, round(im.height * SZER / im.width)), Image.BICUBIC)
    a = np.array(im.convert("L"), dtype=np.float32)
    kandydaci = np.where((a[:, X0:X0 + 30] > 190).mean(axis=1) > 0.8)[0]
    if not len(kandydaci):
        raise ValueError("nie widzę znacznika nad siatką (post musi zajmować całą szerokość ekranu)")
    gora = int(kandydaci[0])
    zm = a[:, X0 + 40:X1 - 40].std(axis=1) > 20
    i, dol, cisza = gora + 60, None, 0
    while i < len(zm):                              # koniec kropek = pierwsza przerwa pod kaflem
        cisza = 0 if zm[i] else cisza + 1
        if cisza >= 8:
            dol = i - cisza + 1
            break
        i += 1
    dol = dol or len(zm)
    im = im.crop((0, gora, SZER, dol)).resize((SZER, ROWS * K + 40), Image.BICUBIC)
    out = Image.new("RGB", (SZER, WYS))
    out.paste(im, (0, Y0 - 40))
    return out


def symbole(im, nr):
    a = np.array(im, dtype=np.float32)
    Y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    sr = Y[Y0 + 1:Y0 + ROWS * K:K, X0 + 1:X0 + COLS * K:K][:ROWS, :COLS]
    yc, yt = jasnosci_palety(nr)
    dc = np.abs(sr[..., None] - np.array(yc)).argmin(-1)
    dt = np.abs(sr[..., None] - np.array(yt)).argmin(-1)
    s = np.where(maska_cyfry(nr), dc, dt).reshape(-1)
    return np.stack([(s >> 1) & 1, s & 1], axis=1).reshape(-1).astype(np.uint8)


def dekoduj_kafel(obraz):
    im = wyrownaj(obraz)
    n = NAGLOWEK.size * 8
    for nr in range(1, 7):                          # maska zależy od cyfry: próbujemy, aż nagłówek się zgodzi
        b = symbole(im, nr)
        glos = (b[:n * KOPIE_NAGLOWKA].reshape(KOPIE_NAGLOWKA, n).sum(0) * 2 > KOPIE_NAGLOWKA).astype(np.uint8)
        h = NAGLOWEK.unpack(np.packbits(glos).tobytes())
        if h[0] == MAGIA and h[2] == nr:
            _, _, _, _, razem, start, ile, *skale = h
            off = n * KOPIE_NAGLOWKA
            w = np.packbits(np.concatenate([np.zeros((ile, 8 - BITY), np.uint8),
                                            b[off:off + ile * BITY].reshape(ile, BITY)], axis=1), axis=1).reshape(-1)
            return nr, razem, start, w, skale
    raise ValueError("nie rozpoznałem kafla")


def odkwantyzuj(w, skale):
    out, p = {}, 0
    for nazwa, k, s in zip(NAZWY, KSZTALTY, skale):
        n = int(np.prod(k))
        out[nazwa] = ((w[p:p + n].astype(np.float64) - (POZIOMY - 1) / 2) * s).reshape(k)
        p += n
    return out


# ── sam model: 15 linii ───────────────────────────────────────────────────────────────────────────
def fiszka(m, x):
    """x = wektor tekstu z EmbeddingGemma (768 liczb). Wynik: pewność dla 17 tematów."""
    h = np.maximum(0, x @ m["W1"] + m["b1"])
    h = np.maximum(0, h @ m["W2"] + m["b2"])
    h = np.maximum(0, h @ m["W3"] + m["b3"])
    return 1 / (1 + np.exp(-(h @ m["W4"] + m["b4"] + x @ m["S"] + m["s"])))


def wektor_ollama(tekst, model="embeddinggemma-2:270m", adres="http://127.0.0.1:11434"):
    """Wektor tekstu z lokalnej Ollamy (EmbeddingGemma 2, 270M; nazwa modelu jak u Ciebie w `ollama list`),
    format dokumentu jak przy treningu: „title: … | text: …”."""
    tytul, _, tresc = tekst.partition("|")
    cialo = json.dumps({"model": model,
                        "input": [f"title: {tytul.strip()} | text: {tresc.strip() or tytul.strip()}"]}).encode()
    req = urllib.request.Request(adres + "/api/embed", cialo, {"Content-Type": "application/json"})
    return np.array(json.load(urllib.request.urlopen(req, timeout=120))["embeddings"][0], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description="Fiszka z Instagrama: model AI ze zrzutów ekranu")
    ap.add_argument("zrzuty", nargs="+")
    ap.add_argument("--wyjscie", default="fiszka_z_instagrama.npz")
    ap.add_argument("--tekst", help="otaguj tekst „tytuł | treść” (wymaga Ollamy z EmbeddingGemma 2)")
    ap.add_argument("--model-wektorow", default="embeddinggemma-2:270m", help="nazwa EmbeddingGemma w Twojej Ollamie")
    a = ap.parse_args()
    w, skale, mam = None, None, []
    for p in a.zrzuty:
        nr, razem, start, czesc, skale = dekoduj_kafel(Image.open(p))
        if w is None:
            w = np.full(razem, (POZIOMY - 1) // 2, dtype=np.uint8)
        w[start:start + czesc.size] = czesc
        mam.append(nr)
        print(f"{p}: kafel {nr}/6 (cyfra {CYFRY[nr - 1]}), wagi {start + 1}–{start + czesc.size}")
    brak = sorted(set(range(1, 7)) - set(mam))
    print("brakuje kafli:", brak or "żadnego", "— brakujące wagi = 0" if brak else "")
    model = odkwantyzuj(w, skale)
    np.savez(a.wyjscie, **model)
    print("model zapisany:", a.wyjscie, f"({sum(v.size for v in model.values())} parametrów)")
    if a.tekst:
        p = fiszka(model, wektor_ollama(a.tekst, a.model_wektorow))
        top = [(TEMATY[i], round(float(p[i]), 2)) for i in np.argsort(-p)[:3] if p[i] >= 0.5]
        print("tematy:", top or "żaden temat nie przekroczył 0,5")


if __name__ == "__main__":
    sys.exit(main())

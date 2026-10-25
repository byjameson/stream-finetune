"""How many tokens does the same content cost in different languages?

This matters for the "small GPU" story more than any tokenizer architecture trend:
if Turkish needs 1.6x the tokens of English for the same text, a 6 GB card buys 1.6x
fewer Turkish words at the same seq length.

Run: python scripts/tokenizer_cost.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer

SAMPLES = {
    "en": (
        "A small graphics card can still fine-tune a large model if the weights are "
        "streamed from disk instead of being held in memory. The trick is to keep only "
        "one transformer block on the GPU at a time, and to recompute the activations "
        "during the backward pass. The cost is disk bandwidth, and the benefit is that "
        "the model no longer has to fit in VRAM or in RAM. This changes who can train: "
        "a student with a laptop can run the same experiment that used to need a rented "
        "data-centre GPU. The measurements in this project come from one machine, and "
        "every number is reported with the configuration that produced it."
    ),
    "tr": (
        "Küçük bir ekran kartı, ağırlıklar bellekte tutulmak yerine diskten akıtıldığında "
        "hâlâ büyük bir modeli ince ayar edebilir. İşin püf noktası, GPU'da aynı anda yalnızca "
        "bir transformer bloğu tutmak ve geri yayılım sırasında aktivasyonları yeniden "
        "hesaplamaktır. Bedeli disk bant genişliğidir; kazancı ise modelin artık VRAM'e veya "
        "RAM'e sığmak zorunda olmamasıdır. Bu, kimin eğitim yapabileceğini değiştirir: dizüstü "
        "bilgisayarı olan bir öğrenci, eskiden kiralık bir veri merkezi GPU'su gerektiren "
        "deneyi çalıştırabilir. Bu projedeki ölçümler tek bir makineden geliyor ve her sayı "
        "onu üreten yapılandırmayla birlikte raporlanıyor."
    ),
    "de": (
        "Eine kleine Grafikkarte kann ein großes Modell immer noch feinabstimmen, wenn die "
        "Gewichte von der Festplatte gestreamt werden, statt im Speicher zu liegen. Der Trick "
        "besteht darin, jeweils nur einen Transformer-Block auf der GPU zu halten und die "
        "Aktivierungen während des Backward-Passes neu zu berechnen. Der Preis ist die "
        "Festplattenbandbreite, der Gewinn ist, dass das Modell weder in den VRAM noch in den "
        "RAM passen muss. Das ändert, wer trainieren kann: Ein Student mit einem Laptop kann "
        "dasselbe Experiment ausführen, für das früher eine gemietete Data-Center-GPU nötig war."
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)

    rows = {}
    for lang, text in SAMPLES.items():
        words = len(text.split())
        ids = tok(text, add_special_tokens=False).input_ids
        rows[lang] = {"words": words, "tokens": len(ids),
                      "tokens_per_word": round(len(ids) / words, 3)}
    en = rows["en"]["tokens_per_word"]
    for lang, r in rows.items():
        r["relative_to_en"] = round(r["tokens_per_word"] / en, 2)
    print(json.dumps({"model": a.model, "rows": rows}, indent=1))
    print("\nAt the same seq budget, the number of words you can fit:")
    for lang, r in rows.items():
        print(f"  {lang}: {round(512 / r['tokens_per_word'])} words at seq 512 "
              f"({round(1024 / r['tokens_per_word'])} at seq 1024)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

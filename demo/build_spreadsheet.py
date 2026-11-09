"""Build an SMS spam classifier as an Excel workbook.

The model is the same 160-word Complement NB as the QR demo; the weights are in a
`model` sheet and the `classify` sheet scores a text with SUMPRODUCT/SEARCH formulas.
No macros, no Python at inference time.

    python demo/build_spreadsheet.py
"""
import re
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT / "research/results/sms_spam.tsv"
OUT = ROOT / "demo/spreadsheet.xlsx"
VOCAB = 160


def load():
    rows = []
    for line in TSV.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        if label in ("ham", "spam") and text.strip():
            rows.append((text.strip(), label))
    return rows


def formula_accuracy(rows, vocab, weights):
    """Approximate the workbook: SEARCH is a case-insensitive substring match."""
    correct = 0
    for text, label in rows:
        low = text.lower()
        scores = []
        for c in range(weights.shape[0]):
            scores.append(sum(weights[c, i] for i, w in enumerate(vocab) if w in low))
        pred = ["ham", "spam"][int(np.argmax(scores))]
        correct += pred == label
    return correct / len(rows)


def main():
    rows = load()
    x = [t for t, _ in rows]
    y = [l for _, l in rows]
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    vec = CountVectorizer(binary=True, max_features=VOCAB, token_pattern=r"(?u)\b\w\w+\b")
    Xtr = vec.fit_transform(xtr)
    clf = ComplementNB(alpha=1.0).fit(Xtr, ytr)
    vocab = list(vec.get_feature_names_out())
    weights = clf.feature_log_prob_.astype(np.float32)
    acc = accuracy_score(yte, clf.predict(vec.transform(xte)))
    f_acc = formula_accuracy(list(zip(xte, yte)), vocab, weights)
    print(f"vocab={len(vocab)} float_acc={acc:.4f} formula_acc={f_acc:.4f}")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "classify"
    ws["A1"] = "SMS text"
    ws["B1"] = "WINNER!! You have been selected to receive a £900 prize reward! Call 09061701461 now to claim."
    ws["A2"] = "ham score"
    ws["B2"] = ('=SUMPRODUCT(--ISNUMBER(SEARCH(model!$A$2:$A$%d,$B$1)),'
                'model!$B$2:$B$%d)' % (len(vocab) + 1, len(vocab) + 1))
    ws["A3"] = "spam score"
    ws["B3"] = ('=SUMPRODUCT(--ISNUMBER(SEARCH(model!$A$2:$A$%d,$B$1)),'
                'model!$C$2:$C$%d)' % (len(vocab) + 1, len(vocab) + 1))
    ws["A4"] = "verdict"
    ws["B4"] = '=IF(B2>=B3,"ham","spam")'
    ws["A6"] = "How to use"
    ws["A6"].font = Font(bold=True)
    ws["A7"] = "Replace B1 with an SMS. B2/B3 are log-probability sums; B4 is the verdict."
    ws["A8"] = "The formula uses SEARCH (substring, case-insensitive), not the exact tokenizer."
    ws["A9"] = "Formula accuracy on the held-out split: %.1f%% (Python model: %.1f%%)." % (
        f_acc * 100, acc * 100)
    ws["B1"].alignment = Alignment(wrap_text=True, vertical="top")
    ws["B4"].font = Font(bold=True, size=14)
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 90
    ws.freeze_panes = "A2"

    m = wb.create_sheet("model")
    m.append(["word", "ham_weight", "spam_weight"])
    for c in range(3):
        m.cell(row=1, column=c + 1).font = Font(bold=True)
        m.cell(row=1, column=c + 1).fill = PatternFill("solid", fgColor="DDDDDD")
    for i, word in enumerate(vocab):
        m.append([word, round(float(weights[0, i]), 5), round(float(weights[1, i]), 5)])
    m.column_dimensions["A"].width = 18
    m.column_dimensions["B"].width = 14
    m.column_dimensions["C"].width = 14
    m.freeze_panes = "A2"

    r = wb.create_sheet("readme")
    r.append(["AI in a spreadsheet"])
    r["A1"].font = Font(bold=True, size=16)
    r.append([])
    r.append(["Model", "Complement Naive Bayes, 160 words, int8-free float weights"])
    r.append(["Training data", "SMS Spam Collection, 5,574 rows; 80/20 split"])
    r.append(["Held-out accuracy", "%.4f (Python), %.4f (spreadsheet formula)" % (acc, f_acc)])
    r.append(["Built by", "stream-finetune / Tiny AI Maker"])
    r.append([])
    r.append(["The classify sheet scores words found in B1. This is a demo of a model"])
    r.append(["small enough to live in a workbook; it is not a production SMS filter."])
    r.column_dimensions["A"].width = 22
    r.column_dimensions["B"].width = 80

    wb.calculation.fullCalcOnLoad = True
    wb.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

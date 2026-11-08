# 0.1 bit araştırması — özet (TR)

**Soru:** Bir modeli 0.1 bit/parametreye kadar sıkıştırıp kaliteyi koruyabilir miyiz?

**Kısa cevap:** Evet — ama "modeli" değil, "davranışı" sıkıştırırsan. Aynı modelin
ağırlıklarını 0.1 bite indirmek modeli öldürüyor. Dar bir davranışı küçük bir öğrenciye
aktarmak, 0.1 bitin çok altına iniyor.

## Ölçülen duvar: quantization

Qwen2.5-0.5B, WikiText-2 test, 16.384 token, grup 64 min-max:

| bit | CE | PPL | boyut |
|---:|---:|---:|---:|
| 16 | 2.782 | 16,2 | 988 MB |
| 4 | 2.994 | 20,0 | 247 MB |
| 3 | 3.963 | 52,6 | 185 MB |
| 2 | 13.062 | 470.679 | 124 MB |
| 1 | 34.066 | 6,2e14 | 62 MB |

4-bit neredeyse bedava; 3-bit PPL'i 3,3× yapıyor; 2-bit uniform tahminden (CE 11,93)
kötü. Post-hoc quantization 3–4 bitte bitiyor.

## 0.1-bit VQ: model ölüyor

Her tensörde 32 ağırlık → 9 prototip (0,099 bit/ağırlık), 6,31 MB:

- CE 12,48 · PPL 262.390
- Aynı 6 MB'ta bigram: CE 6,63 · PPL 757
- Yani 0.1-bit 494M model, 6 MB'lık bir bigram'dan **346× kötü**.

## Damıtma kapısı: AG News

Öğretmen: Qwen2.5-1.5B-Instruct, 6.000 eğitim + 400 test, 4 sınıf.
Öğrenci: TF-IDF + LR, sadece öğretmen etiketleriyle.

| öğrenci | boyut | doğruluk | bit/öğretmen-param |
|---|---:|---:|---:|
| 2k özellik | 135 KB | %73,3 | 0,00074 |
| **5k** | **340 KB** | **%74,3** | **0,00186** |

Öğretmen %75,25; fark 1 puan, McNemar p = 0,689 → istatistiksel olarak yok.
**0,1 bitin 54× altı.**

Aynı etiketlerle BERT-tiny (4,39M): 4-bit 2,19 MB, %76,75 (0,0117 bit/param); 3-bit
1,64 MB, %77,25; 2-bit 1,10 MB, %65,5. Yani sonuç TF-IDF'e özgü değil.

## Kapasite duvarı: SST-2

Öğretmen %94,25. Aynı BERT-tiny 4,4M, aynı 0,0117 bit/param: **%71,75 → -22,5 puan**.
TF-IDF en iyi %71,8 (-22,5). DistilBERT 66M: 4-bit %83,8 (-10,5), 2-bit %73,5 (-20,8).
0,1 bit/param bu görevde yetmiyor; parametre sayısı duvar.

## Yapılandırılmış generasyon: tarih → ISO

Öğretmen 1,5B (500 testte %100 doğru). Öğrenci: 1,35M char-level seq2seq.

- bf16 2,70 MB: %99,8 → 0,0144 bit/param
- 4-bit 0,67 MB: %99,8 → **0,0036 bit/param**

## Açık uçlu generasyon: en zor rejim

Dolly `closed_qa`, 1.000 eğitim / 300 test, öğretmen 1,5B (F1 0,4126). T5-small 60,5M:

| öğrenci | boyut | F1 (referans) | bit/param |
|---|---:|---:|---:|
| bf16 | 121 MB | 0,380 | 0,645 |
| 4-bit | 30,3 MB | 0,256 | 0,161 |
| 3-bit | 22,7 MB | **0,0001** | 0,121 |
| 2-bit | 15,1 MB | **0,000** | 0,081 |

İkinci 4-bit koşu 0,273 verdi; 3-bit ve 2-bit tamamen ölü. Generasyon eşiği 3 bit:
0,1 bit bütçesi burada çok uzak. Yapılandırılmış generasyon (tarih) 0,0036'da çalışırken,
açık uçlu generasyon 0,121'de bile ölü.

## Kontrollü karşılaştırma (asıl bulgu)

Aynı öğrenci (BERT-tiny 4,4M), aynı 0,0117 bit/param:

| görev | öğrenci | öğretmen | fark |
|---|---:|---:|---:|
| AG News (4 sınıf) | %76,75 | %75,25 | **+1,5** |
| DBPedia (14 sınıf) | %91,3 | %92,6 | **-1,3** |
| SST-2 (2 sınıf) | %71,75 | %94,25 | **-22,5** |
| QNLI (2 sınıf) | %59,8 | %81,3 | **-21,5** |
| MNLI (3 sınıf) | %36,8 | %84,7 | **-47,9** |

Aynı mimari, aynı byte, 48 puanlık aralık. **Bit tabanı davranışın özelliği.**

Ucuz teşhis: ham doğruluk farkı karşılaştırılamaz (MNLI -47,9 ama şans %33; QNLI -21,5
ama şans %50). Şansla normalize et: `C = (doğrusal − şans) / (öğretmen − şans)`,
`S = (öğrenci − şans) / (öğretmen − şans)`.

AG News (C 0,98 → S 1,03), DBPedia (0,99 → 0,99), SST-2 (0,49 → 0,49), QNLI (0,21 → 0,31),
QQP (0,21 → 0,49), MNLI (0,18 → 0,07). Fit: **S = 0,92 C + 0,09, R² = 0,885** (6 görev).
Beş görevde S ≈ C güçlü görünüyordu (R² = 0,965); **QQP bozdu**: C 0,21 ama S 0,49
(3 tohum ortalaması; 0,45–0,53). Transformer, TF-IDF'in göremediği paraphrase yapısını
yakalıyor. **C bir alt sınır, öngörücü değil.** Sıradaki adım 10 görev ve çok özellikli model.

## Dürüst konum

- Damıtma algoritması yeni değil (Hinton 2015, DistilBERT, TinyBERT…).
- Yeni olan: "bit/öğretmen-parametresi" metriği, quantization vs distillation'ın aynı
  test setinde karşılaştırılması, görev bağımlılığının ölçülmesi, negatif sonuçlar.
- Devrim için sıradaki adım: görev entropisinden minimum byte'ı önceleyen bir yasa
  (behavioral rate-distortion) veya AutoDistill'in otomatikleştirilmesi.

## Dosyalar

- `docs/BEHAVIORAL_BITS.md` — tam not (EN)
- `research/ARTICLE.md` — dev.to taslağı (EN)
- `research/autodistill.py` — bütçe ver, en küçük geçen öğrenciyi bul
- `research/results/` — ölçüm JSON'ları
- `docs/figures/` — üç figür

## MNLI: kapasite duvarı

3B-Instruct (4-bit) öğretmen %84,7; aynı BERT-tiny 4,4M öğrenci %36,8 (şans %33,3).
DistilBERT 66M aynı etiketlerde 4-bit %57,1 (0,179 bit/param, -27,6) ama 2-bit %35,8
(0,089, -48,9). 0,1 bit bütçesinde ikisi de şans civarı: MNLI daha çok parametre **ve**
daha çok etiket istiyor. Kontrollü karşılaştırma: +1,5 / -21,5 / -22,5 / -47,9.
(1.5B öğretmen önce contradiction'ı %3,3 seçip çökmüştü; 3B ile düzeldi.)

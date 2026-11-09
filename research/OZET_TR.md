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

Öğretmen %75,25; fark 1 puan, McNemar p = 0,689. Ancak tek taraflı %95 alt sınır
**-4,1 puan**; yani 2 puanlık non-inferiority kanıtlanmıyor, sadece “fark tespit
edilemedi” deniyor. **0,1 bitin 54× altı.**

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
- **93 baytlık program:** %100 exact → yalnız kod **4,96e-7 bit/param** (paylaşılan
  752 KB `dateutil` ile 0,0040). 1,35M seq2seq'ten 7.200× küçük ve daha doğru.

**Temsil, sıkıştırmadan önemli:** deterministik bir string eşlemesi için doğru öğrenci
kod; sinir ağı 7.000× gereksiz.

**341 baytlık prompt:** paylaşılan 1,5B taban + 4 örnek = AG News %73,75 (340 KB öğrenci
%74,25). Marjinal 1,82e-6 bit/param; tabanı sayarsan 16 bit/param. **0,1 bit bir muhasebe
sorusu.**

**SST-2'de 294 baytlık prompt: %93,75** (340 KB öğrenci %71,75, öğretmen %94,25). 1.156×
küçük ve 22 puan daha iyi; taban duyguyu zaten biliyor, prompt sadece işaret ediyor.

**Pratik 0,1-bit eseri adapter:** Qwen2.5-3B LoRA (7,37M parametre, 60 SFT adımı)
29,6 MB fp32 = **0,077 bit/taban-param**; 14,8 MB fp16 = **0,038**. Taban paylaşılır;
adapter yetenek ekler. Ama **önce prompt:** aynı tabanda 441 baytlık 4-shot SST-2 promptu
%94,0 (adapter %93,75), 242 baytlık AG News promptu %61,75 (adapter %62,25). Birkaç örnek
görevi anlatabiliyorsa prompt ~10.000× daha küçük ve eşit; adapter promptla ifade
edilemeyen davranış için.

Adapter tam rank: **SVD rank 8, güncellemenin %35,8'ini kaybediyor**; rank indirme işe
yaramıyor. Quantization: **8-bit 8,56 MB = 0,0222 bit/param** (%0,53 hata); 4-bit 4,88 MB =
0,0126 (%9,08). **870 Dolly satırı**, masked loss: taban 2,3752, orijinal 2,3403, 8-bit 2,3401
(p = 0,81; ölçülebilir kayıp yok), 4-bit 2,3420 (+0,0018, p = 0,001; kazancın %5'i).
İkinci görev **SST-2** (2.000 eğitim / 400 test): **%71,25 -> %93,75**; 8-bit ve 4-bit
adapter ikisi de %93,75'te kalıyor, loss 0,7940 / 0,7942 (orijinal 0,7930; p = 0,50 /
0,41). Üçüncü görev **AG News** (4 sınıf, 6.000 / 400): %2,50 (çıplak prompt takip
edilmiyor) -> **%62,25**; 8-bit ve 4-bit %62,00. Güvenli sub-0.1 eser 8-bit; 4-bit maliyeti
göreve bağlı.

Inference: taban **11,79 tok/s**, LoRA yolu **8,75 tok/s** (~%26 yavaş), bf16 LoRA yolu
**9,13 tok/s** (~%22 yavaş); adapter 0,115 s'de yükleniyor. Bu bir **depolama/geçiş
kazancı, token hızı kazancı değil.** bf16 tabana merge edilirse sıfır ek yük; paketlenmiş
4-bit tabana merge burada desteklenmiyor.

**Veri de eser olabilir:** 135 KB'lık coreset (500 örnek) yükleme anında %70,5; 270 KB
(1.000 örnek) %78,0 — 340 KB modelle istatistiksel olarak aynı. Eğitim prosedürü paylaşılır;
depolama ve hesaplama birbirinin yerine geçer.

Merdiven: program (93 B + çalışma zamanı) → prompt (341 B + taban) → **adapter (4,88 MB
4-bit + taban, 0,0126 bit/param)** → coreset (135 KB + eğitim prosedürü) → doğrusal öğrenci
(340 KB) → küçük transformer (2,2 MB) → 4-bit model (30 MB). Alt basamaklar aynı nesnenin
daha iyi sıkıştırılması değil, farklı nesneler. Devrim, basamağı seçmek.

**Ne kullanmalı:**
- deterministik → **program** (93 B)
- taban zaten kurulu ve yapabiliyor → **prompt** (294–341 B)
- taban kurulu, görev dışında → **adapter** (4,88 MB, 0,0126)
- taban yok, az veri → **coreset** (135 KB)
- taban yok, anahtar kelime → **doğrusal** (340 KB)
- taban yok, semantik → **küçük transformer** (2,2 MB)
- genel yetenek → **4-bit model** (30 MB+)

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
QQP (0,21 → 0,49), MNLI (0,18 → 0,07), 20 Newsgroups (0,60 → 0,42), Emotion (0,55 → 0,59).
Fit: **S = 0,90 C + 0,07, R² = 0,827** (8 görev). Emotion çizgiye oturuyor. Beş görevde
S ≈ C güçlü görünüyordu
(R² = 0,965). **QQP yukarı bozdu** (C 0,21, S 0,49): transformer paraphrase yapısını
yakalıyor. **20NG aşağı bozdu** (C 0,60, S 0,42): zayıf öğretmen (%29,5) ve 20 sınıf için
sınıf başına 100 etiket az. **C bir alt sınır, öngörücü değil.** Sıradaki adım 10 görev ve
çok özellikli model.

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

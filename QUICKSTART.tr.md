# Hızlı Başlangıç (Türkçe)

Sadece `streamft` komutları. Repo, script, bulut hesabı gerekmez.

## 0. Kurulum

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

Her yerde:

```bash
pip install streamft
streamft doctor
```

`doctor` torch/CUDA/bitsandbytes sürümlerini ve boş VRAM'i yazar. GPU'suz da devam
edebilirsin (bkz. §4).

## 1. Sığar mı?

```bash
streamft fit --model Qwen/Qwen2.5-14B-Instruct --gpu 6 --seq 512
```

```text
13.99B params, 48 layers
NF4 blocks ~7.5 GB + bf16 head ~1.6 GB + reserve 0.9 GB (embedding stays on CPU)
-> whole model needs ~9.9 GB on 6 GB
verdict: needs streaming
measured reference: Qwen2.5-14B (model_type qwen2, layers 48, hidden 5120) on RTX 3050 6 GB
  -> 6 block(s), 3.86 GB peak, 5.86 s/step at seq 512
note: SFT: one Dolly row per step, median ~134 tokens (seq is a cap on this path)
```

Bellek kısmı `config.json`'dan aritmetik. **"Sığar" bir bütçe kontrolüdür, garanti değil** —
aktivasyonlar ve geçici ayırmalar tam modellenmiyor. Hız yalnızca ölçülmüşse yazılır.

Çıkarım için:

```bash
streamft fit --model Qwen/Qwen2.5-7B-Instruct --gpu 6 --seq 2048 --mode inference
```

## 2. Hazırla ve eğit (SFT)

```bash
streamft prepare --model Qwen/Qwen2.5-0.5B --save-quant checkpoints/q05
streamft train --model-dir checkpoints/q05 --sft --steps 20 --seq 512 --tag ilk
```

- `--steps N` = **N optimizer adımı**. Prompt maskesi tutmayan satırlar baştan elenir ve
  `sft_skipped` olarak sayılır.
- Her koşu `results/results.jsonl` dosyasına bir satır yazar: loss'lar, adım süresi, tepe
  VRAM, `grad_layers`, adaptör hash'i, GPU ve lisans.
- Adaptör sadece `--save checkpoints/benim_adaptorum` verirsen diske yazılır.

## 3. Eski kart (GTX 1060/1660 — bf16 yok)

```bash
streamft train --model-dir checkpoints/q7 --sft --steps 300 --seq 512 \
  --compute-dtype fp16 --blocks-on-gpu 1 --vram-hard-gb 5.3
```

4 GB kartta `--blocks-on-gpu 1 --seq 512 --vram-hard-gb 3.4` ile başla. Bu satırlar
henüz ölçülmedi; koşarsan JSON satırını gönder, tabloya girsin.

## 4. GPU yok — CPU'da eğit

```bash
streamft cpu --model Qwen/Qwen2.5-0.5B --steps 50 --seq 128 --save checkpoints/cpu
```

Ryzen laptopta ölçüm: 0.5B, seq 128, **1.73 s/adım** (50 adım medyanı), eval 2.934 → 2.835.
Model RAM'e sığmalı; 4-bit ve streaming yok. Amaç "hiç yapamamak" değil.

## 5. Veri yok — kendin üret

```bash
streamft synth --model Qwen/Qwen2.5-1.5B-Instruct --n 200 \
  --task "pratik mutfak ipuçları" --out data/synth.jsonl
streamft train --model-dir checkpoints/q7 --sft --dataset data/synth.jsonl \
  --train_size 150 --eval_size 20 --steps 200 --seq 512
```

Şablon çöpü (`<instruction>`) ve tekrarlar filtrelenir. 1.5B ile ölçüm: 30 deneme → 13
geçerli çift (%43), ~5 s/deneme.

## 6. İşe yarıyor mu?

```bash
streamft eval --model Qwen/Qwen2.5-1.5B-Instruct
```

20 prompt (10 İngilizce + 10 Türkçe) ile tekrar oranı, reddetme oranı, anahtar kelime
isabeti ve tok/s ölçer. API yok, hakem model yok. Kendi promptların:

```bash
streamft eval --model ./merged --prompts benim_promptlarim.jsonl --max-new 96
```

Satır biçimi: `{"prompt": "...", "expect": ["kelime"]}` (`expect` isteğe bağlı).

## Sorun giderme

- **Başlangıçta CUDA OOM**: `--blocks-on-gpu 1`, `--vram-hard-gb 5.3`.
- **İlk adım yavaş**: ağırlıklar diskten okunur; sayılar sıcak medyandır.
- **`fit` "sığar" dedi ama taştı**: aktivasyonlar modellenmiyor; seq'i veya blok sayısını azalt.

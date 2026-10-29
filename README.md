# Halı Saha Analizi — MVP 0.1

Bu ilk prototip **oyuncu tespiti + geçici ID takibi + ham piksel koordinatları** üretir. Henüz eğitilmiş özel halı saha modeli, kalıcı oyuncu kimliği, takım rengi, güvenilir top takibi, pas/şut sınıflandırması veya gerçek mesafe hesabı içermez.

## Kurulum

Python 3.10–3.12 önerilir.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
python run_tracking.py --video ilk_15_saniye.mp4 --output results --model yolo11n.pt
```

İlk çalıştırmada Ultralytics model ağırlığını indirebilir; internet erişimi gerekir. Çıktılar `results/tracked.mp4` ve `results/detections.csv`.

`--tracker bytetrack.yaml` seçeneği ile alternatif takip yöntemini deneyebilirsiniz. `--imgsz 1280` küçük oyuncular için denenebilir (daha yavaş). Takip sürekliliği için `--stride 1` önerilir.

## İki kamera açısı için plan

- **Orta kenar:** ayrı saha köşeleri, çizgiler ve kamera homografisi.
- **Köşe:** aynı algılama modeli, farklı homografi.
- Her açıdan etiketli kareler birleştirilerek aynı algılama modeli ince ayarlanacak.
- Oyuncu ayak noktaları saha düzlemine taşınacak. Homografi için saha üzerinde bilinen en az dört eşleşen nokta gerekir; geniş açı lens distorsiyonu varsa önce düzeltilecek.
- Top için küçük nesnelere özel veri ve ayrı eğitim gerekebilir. COCO `sports ball` sınıfı tek başına güvenilir değildir.

## Sonraki geliştirme

1. Farklı maçlardan örnek kareleri `side/` ve `corner/` şeklinde etiketlemek.
2. Oyuncu ve top için doğrulama seti oluşturmak; maç bazında train/val ayrımı.
3. Takım rengi, numara, kimlik galerisi ve uzun süreli Re-ID.
4. Kalibrasyon ve hız/mesafe doğrulaması.
5. Topa sahip olma ve pas/şut olay etiketleme.

## Lisans

Ultralytics paketinin ve kullanılacak model ağırlıklarının lisans koşullarını dağıtım/ticari kullanım öncesi kontrol edin.

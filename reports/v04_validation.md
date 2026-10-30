# v0.4 — Ölçüm ve insan incelemesi altyapısı

Bu çalışma bir doğruluk kilometre taşı değildir. İnsan onaylı olay kayıtları hedefi için
etiketleme, ölçüm ve tekrar oynatma altyapısı eklendi. Mevcut top kararları ve oyuncu takip
mantığı değiştirilmedi; yeni dedektör çalıştırılmadı, model eğitilmedi.

## Yapılan kontroller

| Kontrol | Sonuç |
|---|---|
| Python 3.10.11, unittest discover | **55 test geçti**: 31 eski, 24 yeni |
| Kısa örnekte top tekrar oynatımı | 396 aday, 450 kare durumu, 54 kabul |
| Eski sonuçla aday/kare alanları karşılaştırması | **0 fark**, iki önbellek yolunda da |
| Zayıf oyuncu kutusu tanısı | 296 zayıf kimlikli satır; 67 yüksek-IoU çift, 63 kare |
| İnsan etiketi | **0**; doğruluk/IDF1/HOTA hesaplanmadı |
| Hazırlanan ham inceleme verisi | 360 PNG, 13 klip, toplam 1.350 çözümlenmiş klip karesi |
| Olay hipotezi koşusu | 0 aday; gerçek olay sayısı ölçülmedi |
| JavaScript | Son arayüz kodu sözdizimi kontrolünden geçti |
| HTTP / kayıt | Yerel servis, kaynak kontrolü, revizyonlar ve sentetik API testleri geçti |
| Tarayıcı görsel kontrolü | **Tamamlanamadı:** Computer Use uygulama erişimi zaman aşımına uğradı |

55 testin tamamı sentetik veya taşıma/uyumluluk kontrolüdür. Sentetik testlerdeki insan kaynak
alanları yalnız geçici test dizinlerinde kullanılır; gerçek inceleme paketine insan etiketi
yazılmadı. Test günlüğü: `results/v04_tests.txt`.

Yeni testler özellikle görüntü ve CSV karar eşleşmesini, mevcut çevrimiçi CLI'nin sahte
dedektörle kanıt ihracını, innovation değerinin yeniden hesaplanmasını, bilinmeyen/eski özellik
tanımının reddini, AI etiketlerinin dışlanmasını, aday yokluğu/seçim hatası ayrımını, gizli
top koordinatı uydurmamayı, eksik kimlik ihracının reddini, revizyonları, belirsiz olay
aralıklarını, bire bir eşleşmeyi, tek kareden sahiplik kurmamayı ve kesit sıfırlamayı sınar.

## Tekrar oynatımın kapsamı

Girdi: `results/v031_final`, 450 kare / 640×360 / yaklaşık 29,97 FPS, tarihî YOLO11n
imgsz 960 / stride 1 çıktısı. O çıktıya ait tespitler yeniden üretilemediği için geriye dönük
model sürümü veya ağırlık SHA256'sı uydurulmadı; yeni çevrimiçi koşular bunları kaydeder.

İki yol tamamlandı:

1. `results/v04_ball_cache`: ham gözlemdeki tam skor/kutular ve kayıtlı arka planla görüntü
   özellikleri tekrar hesaplandı; yalnız kısa video kareleri açıldı, YOLO çalışmadı.
2. `results/v04_ball_csv_cache`: kayıtlı aday oranları CSV'den, kişi/gövde örtüşmesi ham kişi
   kutularından, geometri mevcut yapılandırmadan alındı; görüntü/model açılmadı.

İkinci yolun eski oranları dört ondalığa yuvarlanmış olması sözleşmede açık. Her iki yolun
tekrarı, eski `ball_candidates.csv` alanlarının tamamında ve `ball_track.csv` durumlarında
eşleşti. Bu bütün kameralarda yuvarlama zararsızdır anlamına gelmez; her önbellekte parite
kontrol edilmelidir. Özellik eşiği/geometri değişirse kanıt yeniden kurulmalıdır.

Çıktılar: `results/v04_ball_replay/replay_report.json` ve
`results/v04_ball_csv_replay/replay_report.json`.
Kişi takibi değiştirilmedi. Kullanılan kısa önbellekte 75 sahne-kapsamlı kişi ID'si ve
11 mekânsal geçiş ipucu bulunuyor; bunlar gerçek oyuncu/ID değişimi doğruluğu değildir.

Tek bir sıcak ortam CSV tekrarı, dosya yazma ve referans karşılaştırması dâhil yaklaşık
**0,417 saniye** sürdü; tracemalloc tepe değeri **2.397.256 bayt**.
Bu yalnız Python tahsisleri; işlem RSS/GPU belleği, dedektör hızı veya genel performans
garantisi değildir. Artefakt: `results/v04_replay_benchmark/benchmark.json`.

## İnceleme paketi ve etiket sınırı

`results/review_v04/dataset.json` kaynak kareleri, dosya özetlerini, pencereleri, ayrımı ve
tarihî parametreleri saklar. Kısa klipte olağan oyun, 170–209 zayıf gözlem örneği ve
325–364 bilinen top hatası/toparlanma örneği seçildi. Montajın 2, 6 ve 8. kesitlerinden
ikişer 40 kare ve tohum 42 ile birer altı saniyelik olay denetim penceresi eklendi.
Kısa örneğin tamamı olay bağlamı için ayrıca ham klip olarak yer alır.

Geliştirme: kısa kaynak ve montaj kesit 2. Değerlendirme: montaj kesit 6/8.
Bu yalnız kesit ayrımıdır; farklı maç ve tekrar görüntüsü bağımsızlığı kanıtlanmadı.
Kısa önbellek stride 1/imgsz 960, uzun tarihî önbellek stride 3/imgsz 640'tır.
İşlenmeyen karelerde öneri olmaması negatif etiket sayılmaz.

Ham örneklerin temas sayfası `contact_sheet.jpg` içindedir. Kamera/geometri çeşitliliğine
ilişkin tek AI gözlemi `visual_review_notes.json` içinde `ai_reviewed` olarak ayrıldı.
Gerçek kare/olay/kimlik etiketleri incelenmedi. Açı isimleri insan doğrulaması bekliyor.
Çıkarılmış PNG'lerde kutu yok; kaynak videonun kendi logo/tarih yazıları korunur.

`evaluation_all.json` insan etiketi sıfır olduğundan metrikleri null bırakır. Ham top
kapsaması ile seçilmiş topun precision/recall/merkez hatası ayrı uygulanmıştır; mevcut
veride bunlara sayı üretilmedi. IDF1/HOTA için referans araç kurulmadı. İhracat
`results/v04_trackeval` altında tüm dokuz kimlik penceresini eksik diye atladı.

## Olaylar, belirsizlik ve açık işler

Sahiplik durumları player/free_ball/contested/unknown; başlatma/bırakma ayrı sürelerle kaynak
zamanındadır. Yakınlık ve hızlı görüntü hareketi doğrulanmamış hipotezdir. Eksik top gözleminde
oyuncu sahipliği tutulmaz; kesitte durum sıfırlanır. Takım/kale bağlamı bilinmiyor.
Pas, şut veya gerçek tackle kararı otomatik verilmedi.

Bu pencerelerdeki sıfır aday sonucu eşiklerin iyi olduğunu veya olay bulunmadığını göstermez.
Rastgele pencereler ve önerilmeyen anlar da incelenmelidir. İnsan onayı olmadan top/pas/şut
doğruluğu yoktur. Belirsiz/yarım bağlam ayrı aralık olarak kaydedilir; tekrar görüntüsü
duplicate_of ile tekilleştirilir. Doğrulanmış takip–insan kimlik eşlemesi olmadan olayın
oyuncu atama doğruluğu da hesaplanmaz.

Kayıtlı 67 kutu çifti bastırılmadı; hepsi hâlâ incelenmemiş örtüşmedir. Aynı veya farklı
insan kutusuna eşleşme tanısı, insan tarafından tamamlanan pencerelerde kullanılabilir.
Kalibrasyonda bağımsız ölçülü doğrulama, takım sınıflandırması ve kalıcı kimlik hâlâ yok.
Tarayıcıda fareyle düzenleme ve oynatma davranışı insan tarafından kontrol edilmeli.
Kayıt dosyaları tek kullanıcı içindir; çok dosyalı işlem çökmesinde revizyon günlüğünden
kurtarma gerekebilir. Etiketlerin yedeği alınmalı.

## Dosya envanteri

Değişen mevcut dosyalar: `ball_tracking.py` (kararları değiştirmeden kanıt/ilişkilendirme
ayrımı), `run_tracking.py` (kanıt ihracı), `detector.py` (model provenansı), `README.md`,
`AGENTS.md`, `reports/v03_validation.md` ve `reports/v031_validation.md`.

Yeni çalışma dosyaları: `ball_cache.py`, `replay_ball.py`, `review_schema.py`,
`prepare_review.py`, `review_store.py`, `review_app.py`, `review_ui.html`,
`evaluate_review.py`, `export_trackeval.py`, `audit_players.py`, `event_candidates.py`,
`test_review_workflow.py`.
Yeni belgeler: `docs/review_workflow.md`, `docs/detector_options.md`,
`reports/history_v01_v031.md` ve bu rapor.

Türkçe bozulmalar bağlamdan düzeltildi; anlamlı P?/BALL? korundu. Tarihî v0.3.1 raporundaki
29 test sayısı 31 olarak düzeltildi. Eski README arşivlendi; güncel README kullanım odaklı.
Baseline, kişi takipçisi, eski TemporalBall, bağımlılıklar ve ağırlıklar değiştirilmedi.

İnceleme kararı bekleyenler: ilk insan etiketleri, açı/tekrar gruplarının doğrulanması,
pilot 5 px toleransının değerlendirme başlamadan kabulü veya açık protokol sürümü değişikliği,
olay tanımlarının ürün beklentisiyle onayı ve ileride proje/bağımlılık lisans kararı.
Dedektör karşılaştırma kapısı, görünür topun adaylarda olmadığı insan etiketiyle ölçülene
kadar açılmadı. Eğitim önerilmedi. Doğrulama sonunda çalışma commit edilmeden incelemeye
sunuldu; sahibinin sonraki açık isteği bu checkpoint'in commit edilmesini ve mevcut dalda
origin'e gönderilmesini yetkilendirdi.

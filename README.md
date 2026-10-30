# Halı Saha Analytics — v0.4 inceleme altyapısı

Hedef: sabit kameralardan amatör futbol görüntülerinde **insan onaylı pas, şut ve top kazanma kayıtları** ve incelenen kesitlere ait istatistikler. Mevcut YOLO11n ve BoT-SORT/ByteTrack hattı korunuyor. Bu sürüm ölçüm/etiketleme ekler; otomatik maç istatistiği doğruluğu kanıtlamaz. Dal: `feature/tracking-v02`; önceki checkpoint: `6b67f79`. v0.4 checkpoint'i sahibinin açık commit/push isteğiyle bu dalda sürümlenir.

## Kullanım

Python 3.10–3.12 ve mevcut ortam kullanılır. Yeni bağımlılık yok. Gerekirse `pip install -r requirements.txt`; `constraints-tested.txt` eski bildirilen sürümleri kaydeder, tam transitif kilit değildir. Yerel ağırlık yoksa Ultralytics indirebilir.

~~~powershell
python run_tracking.py --video ilk_15_saniye.mp4 --output results/new_online
python track_csv.py --detections results/v031_final/raw_detections.csv --output results/new_player_replay
# Haricî CSV: bilinen FPS/kare sayısı boş kareleri korur.
python track_csv.py --detections provided.csv --fps 30 --frame-count 450 --output results/external_replay
~~~

`python run_tracking.py --help` tüm seçenekleri gösterir. Kişi ham tabanı `--person-det-conf 0.10`, güçlü/atanmamış gösterim `--conf 0.25`, top eşiği `--ball-conf 0.15` değerindedir; bunlar evrensel doğruluk eşikleri değildir. Tek tam-kare çıkarım paylaşılır. Ek kırpım deneyi `--ball-search` varsayılan kapalıdır.

## İnsan incelemesi

Hazır yerel paket `results/review_v04/`: **360 ham PNG kare, 13 kısa klip, 1.350 çözümlenmiş klip karesi, 0 insan etiketi**. Açı isimleri insan tarafından doğrulanmadı; uzun videoda yeni YOLO çalıştırılmadı.

~~~powershell
python review_app.py --dataset results/review_v04
# Tarayıcı: http://127.0.0.1:8765
# Yeniden hazırlamak gerekirse farklı, boş bir klasör:
python prepare_review.py --output results/review_new
python event_candidates.py --dataset results/review_v04
python evaluate_review.py --dataset results/review_v04 --split eval
python audit_players.py --dataset results/review_v04 --output results/new_player_audit
~~~

Kutuyu seç/taşı, sağ altından boyutlandır veya sayısal koordinatları düzelt. Eksik topu merkezine tıklayarak ekle; fazla kutuyu reddet. Görünürlük ve inceleyen adını gir, **Bu kareyi inceledim** ile kaydet. `human`, `ai_reviewed`, `proposal` ayrıdır; yalnız insan incelemesi doğruluk hesabına girer. Kişi metriği için tüm görünür kişiler, kimlik metriği için ayrıca kimlik zincirleri tamamlanmalı.

Ham klip/karelerden olayları incele: onayla, reddet, tür/zaman düzelt, belirsiz bırak; önerilmeyen olayı da ekle. Pencere tamamen incelenmeden kaçırılan olay metriği hesaplanmaz. Revizyonlar `review_history.jsonl` içinde önceki değerlerle tutulur. MP4V desteklenmiyorsa ham kare oynatımı kullanılabilir; önizleme gerçek zaman hızını garanti etmez. İlk koşu **0 olay adayı** üretti; bu, görüntülerde 0 olay olduğu anlamına gelmez. Şut adayı onayı otomatik şut sayılmaz.

## Görüntüsüz top tekrarı

~~~powershell
python replay_ball.py build --run results/v031_final --output results/new_ball_cache --legacy-features --feature-version pitch_max_core_v1
python replay_ball.py replay --cache results/new_ball_cache/ball_evidence.csv --output results/new_ball_replay --reference-run results/v031_final
# Alternatif: özellikleri referans + ham karelerden bir kez hesapla; YOLO yok:
python replay_ball.py build --run results/v031_final --output results/new_full_precision_cache
~~~

Yeni çevrimiçi pitch koşuları `ball_evidence.csv` ve `.meta.json` yazar: sürümlü görüntü özellikleri, kişi/gövde örtüşmesi, saha dışı kontrolü ve boş kareler dâhil zaman çizelgesi. `innovation_px` ve takip seçimleri yeniden hesaplanır. Geometri/özellik tanımı değişirse önbellek yeniden kurulmalı. Eski CSV oranları yuvarlanmıştır; eski v0.3 ortalaması ile v0.3.1 merkez maksimumu karıştırılmaz. İki tekrar yolu da **396 aday / 450 kare / 54 kabul / 0 karar farkı** verdi; bu doğruluk değildir.

## Modüller ve sözleşme

| Parça | Görev |
|---|---|
| `baseline_tracking.py` | Değiştirilmemiş v0.1 |
| `detector.py` / `detection_io.py` | Dedektör adaptörü, özgün CSV, gelecek koşular için model/ağırlık özeti |
| `player_tracking.py` / `track_csv.py` | Mevcut takipçi, görüntüsüz oyuncu tekrarı |
| `ball_cache.py` / `replay_ball.py` | Sürümlü top kanıtı ve aynı kararların tekrarı |
| `prepare_review.py` / `review_schema.py` / `review_store.py` | Ham paket, ayrı etiketler, kaynak/revizyon |
| `review_app.py` / `review_ui.html` | Localhost düzenleme ve olay inceleme |
| `evaluate_review.py` / `export_trackeval.py` | İnsan etiketleriyle ölçüm, referans kimlik aracı ihracı |
| `audit_players.py` / `event_candidates.py` | Çift kutu tanısı, doğrulanmamış sahiplik/olay hipotezleri |
| `split_scenes.py` / `compare_ball_filters.py` | Yeniden kullanılan klip ihracı / tarihî filtre karşılaştırması |

Özgün 11 alan korunur: `frame,time_s,track_id,class_name,confidence,x1,y1,x2,y2,foot_x,foot_y`. Takip uzantıları: `object_type,jersey_l,jersey_a,jersey_b,scene_id,pitch_x_m,pitch_y_m,tracking_state`. Ham kimlik -1; haricî kimlik yeniden takipte kullanılmaz. Ayak noktası alt-orta, top için fiziksel temas noktası değildir.

`track_states.csv` oyuncu gözlem/atanmamış/tahmin ve top tahminlerini; `ball_track.csv` kabul edilen top durumlarını içerir. Tahminler ölçüm değildir. `P?` atanmamış tespit, `BALL?` geçici adaydır. `occluded` gerçek örtülme etiketi değildir. Kimlik anahtarı `(source_id,scene_id,object_type,track_id)`; kare sıfırdan, zaman kaynak kare/FPS. Kalibrasyonsuz metrik sütunlar boş; Parquet yok.

Şema, ölçüm ve olay tanımları: [iş akışı](docs/review_workflow.md). Lisanslar ve karşılaştırma kapısı: [dedektör seçenekleri](docs/detector_options.md). Testler ve çıktı kanıtları: [v0.4 raporu](reports/v04_validation.md).

## Kontroller ve vaat sınırı

~~~powershell
python -m unittest discover -v
git diff --check
~~~

İnsan etiketli doğruluk ve bağımsız gerçek kalibrasyon yok. Lab takım sınıfı, takip numarası gerçek kimlik değildir. Şut hızlı topa, top kazanma otomatik tackle'a eşitlenmez. Kesit üzerinden olay zinciri kurulmaz; yarım bağlam dışa aktarılır. Montaj tekrarları insan tarafından tekilleştirilir.

Vaat: **“İncelenen görüntülerde N onaylı pas; şu aralıklar belirsiz.”** Eksiksiz maç toplamı, doğrulanmamış metre/hız/mesafe veya havadaki topun 3B hareketi sunulmaz. Etiketsiz dedektör karşılaştırması, eğitim, lisans seçimi ve bağımlılık değişikliği yapılmadı.

Tarayıcı görsel kontrolü uygulama erişimi zaman aşımı nedeniyle tamamlanamadı. HTTP/API ve sentetik kayıt davranışı test edildi; sürükleme ve tarayıcı oynatımı kullanıcı tarafından gözden geçirilmeli.

Tarihçe: [v0.1–v0.3.1 eski README](reports/history_v01_v031.md), [v0.3](reports/v03_validation.md), [v0.3.1](reports/v031_validation.md). Rutin geliştirmede uzun montaj yerine `results/scene_clips/` ve manifestini kullan; yeni koşuda boş çıktı klasörü seç.

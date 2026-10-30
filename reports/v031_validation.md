# v0.3.1 - etiket kayıpları ve tespit/takip ayrımı

Çalışma `feature/tracking-v02` üzerinde, v0.3 kontrol noktası `b4be422` sonrasında yapıldı. Kullanıcı bu turda hem kişi hem top hattını iyileştirmeyi, ayrıca ham CSV üzerinden model çalıştırmadan kişi takibini denemeyi onayladı. Uzun montaj yeniden YOLO'dan geçirilmedi; doğrulama 450 karelik kısa video ve uzun videodan bilinen iki arızalı kare üzerinde yapıldı.

## Kişi etiketleri neden kayboluyordu?

1. Eski dedektörün kişi güven eşiği 0,25 idi. Takipçinin ikinci aşaması 0,10–0,25 arası gözlemlerle eşleştirme yapabilecek olsa da bu gözlemler ona ulaşmıyordu.
2. Kurulu Ultralytics sürümünün `track.py:on_predict_postprocess_end` callback'i, aktif izler olduğunda sonucu eşleştirilmiş izlerin indisleriyle süzüyor. Yeni ve henüz onaylanmamış izleri de gizleyebiliyor. Bu nedenle güçlü bir ham kutu bulunması her zaman ekranda etiket bulunması anlamına gelmiyordu.
3. Gerçek model kaçırmaları, örtüşme, düşük çözünürlük, kesit değişimi ve hatalı eşleştirme de etiket kaybına yol açabilir. İlk iki düzeltme bunları tamamen çözmez.

Mevcut kısa boşlukların uzamsal incelemesinde, eski bir ID'nin iki gözlemi arasındaki 1–3 karelik boşluklar için kutular doğrusal olarak ara değerlendi. Ham tespitler IoU >0,30 ile karşılaştırıldı. Bu yöntem 73 boşluk-kare örneğinde 55 düşük güvenli ve 18 güçlü yakın tespit buldu. Örneğin eski ID 1 için 183. karede yakın ham tespitin güveni 0,1826; eski ID 10 için 387. karedeki yakın tespit 0,5238 idi. Bu inceleme elle etiketlenmiş kimlik doğrulaması değildir; yakındaki başka bir kişiyle eşleşme olasılığı vardır.

## Değişiklikler

- `detector.py`: YOLO bir adaptörün arkasında. Kişi ve top için ortak tam-kare çıkarımından sonra ayrı güven eşikleri uygulanıyor. Varsayılan kişi tabanı 0,10; güçlü/gösterilecek bağımsız gözlem eşiği 0,25. Dedektör değiştirilirse takipçiye aynı `Detection` kayıtları verilebilir.
- `detection_io.py`: eski 11 sütunlu ve genişletilmiş CSV için ortak giriş. Ham kayıtların ID'si -1; sonradan atanacak kimlik değildir. Ham skor ve koordinatlar tam hassasiyetle tutulur. Geçersiz/NaN kutular reddedilir.
- `player_tracking.py`: mevcut BoT-SORT/ByteTrack arka ucuna dedektörden bağımsız gözlem dizileri verilir. Zayıf gözlemler mevcut izleri kurtarabilir. Eşleşmemiş güçlü tespitler gizlenmez, `P?` olarak gösterilir. Yeni kimlik uydurulmaz. Saha filtresi yapılandırılmış ayak noktası üzerinde eşleştirmeden önce çalışır.
- Kısa boşlukta varsayılan en çok 3 kaynak kare boyunca kesikli gri `predicted` etiket gösterilir. Tahminler `detections.csv` içine yazılmaz ve gözlem sayısına katılmaz. `--label-hold 0` bu davranışı kapatır.
- `track_csv.py`: YOLO ağırlığı veya video gerektirmeden ham CSV'den kişi takibi tekrar oynatılır. İsteğe bağlı aynı ham video yalnızca inceleme videosu/renk özelliği üretir. Boş kareler, FPS, stride ve sahne sınırları metadata ile korunur.
- `calibration.py`: isteğe bağlı gerçek saha boyutları ve homografi verilince görüntü saha dörtgeni türetilebilir. Gerçek örneğe ölçülmüş kalibrasyon verilmediği için burada metre koordinatları üretilmedi.
- `ball_tracking.py`: alternatif top hipotezleri aktif yanlış hipotezin kapısında kalıcı olarak yok edilmez. Tutarlı yeni adaylar tekrar gözlemlenince devralabilir. Ani duruş/dönüşte yalnızca ileri tahmine değil son gözleme yakınlığa da bakılır. Kişiyle örtüşme yumuşak bir sıralama cezasıdır; gerçek topun oyuncuyla örtüşmesi yasaklanmaz.
- `pitch_context.py`: beyaz noktanın büyük bir kutuda seyrelmesini azaltmak için hem bütün kutu hem merkez bölge incelenir. Beyazlık ve değişim oranları bu iki bölgenin maksimumudur. Eski v0.3 dosyalarındaki bütün-kutu oranlarıyla aynı tanım değildir.
- `ball_detection.py`: periyodik bölgesel tarama ve aktif/alternatif ROI deneyi eklendi fakat varsayılan **kapalı**. Mevcut klipte ek maliyete karşı doğrulanmış ROI tespit kazancı oluşmadı.

## Kontrollü kişi karşılaştırması

Aynı kısa video, aynı model, `imgsz=960`, `stride=1`. Ham kişi kayıtları iki takip ayarına da verildi; model yeniden çalıştırılmadı. Güçlü eşikli karşılaştırma ham 0,25 ve üzerindeki gözlemleri kullanır; yeni ayar 0,10 tabanını kullanır. Saha maskesi verilmediğinden bu çalışmada saha dışı filtreleme sayısı 0'dır.

| Ölçüm | 0,25 tabanı, tahmin etiketi yok | Yeni 0,10 tabanı |
| --- | ---: | ---: |
| Takipçiye verilen kişi kutusu | 6.552 | 8.667 |
| ID atanmış gözlem | 6.146 | 6.446 |
| 0,25 altındaki ID'li gözlem | 0 | 296 |
| Güçlü ama atanmamış gözlem | 406 | 402 |
| Sahneye göre farklı kişi ID'si | 80 | 75 |
| Olası uzamsal ID geçişi | 18 | 11 |
| Ayrı tutulan tahmini etiket | 0 | 93 |

Yeni `detections.csv` kişi satırları 6.446 ID'li + 402 atanmamış = **6.848** gözlemdir. 93 tahmin bu sayıya dahil değildir. Online çalışma ile CSV-only tekrar oynatmanın 6.848 kişi satırı ilk 11 sütunda **birebir eşleşti**. Deneysel crop çalışmasının ve son varsayılan çalışmanın kişi ham kayıtları da birebir aynıdır; kontrollü karşılaştırma son çalışmaya aynen uygulanır.

Bu sayılar gerçek kimlik başarısı değildir. Daha az ID bazen daha iyi süreklilik, bazen yanlış birleştirme anlamına gelebilir. IDF1/HOTA, kimlik etiketleri olmadan hesaplanmadı. Gözlemler içinde hakem, seyirci ve çift kutular bulunabilir. Oyuncu sayısını 10–14'e zorlayan bir üst sınır eklenmedi; böyle bir kural görünür gerçek kişileri keyfi olarak düşürebilir.

## Top karşılaştırması: küçük ve sınırlı kazanım

Son varsayılan çalışma 396 ham top adayı üretti. `b4be422` commit'indeki özgün top filtresi ve özgün arka plan kuralı aynı kayıtlar üzerinde yeniden oynatıldı: **52** kabul. Yeni filtre: **54** kabul. Varsayılan çalışmada ek crop çıkarımı **0**.

| Kare | Eski v0.3 | Yeni v0.3.1 |
| --- | --- | --- |
| 202 | Görünen top kabul ediliyor | Aynı gözlem korunuyor |
| 331–332 | Oyuncu kıyafeti/ayakkabısı civarı yanlış seçiliyor | Bu hata hâlâ var |
| 333–334 | Doğru adaya rağmen kabul yok | Yeni hipotez görünen topu kabul ediyor |
| 350 | Görünen top kabul ediliyor | Aynı gözlem korunuyor |

Kare durumları: `missing=355`, `observed=54`, `occluded=37`, `tentative=4`. `missing` gerçek top görünür olduğu hâlde kaçırılan kare sayısı değildir; görünürlük etiketi yok. `observed` doğru top garantisi değildir. Top için 15 kısa tahmin etiketi ayrıca tutulur; accepted/detection sayısına dahil değildir. `BALL?` yalnızca mevcut ama onaylanmamış adaydır.

Kısa videoda incelenen sabit beyaz bölgedeki kabul yine 0'dır. Uzun örneklemeli videoda bilinen 78 ve 81. karelerdeki iki yanlış kutu, yalnızca bu kareler ve kayıtlı sahne arka planı okunarak tekrar incelendi: eski bütün-kutu beyaz oranı 0,1528; yeni merkez/bütün-kutu oranı 0,50; görüntü farkı 0. Yeni karar ikisi için de `static_marking`. Uzun videonun tamamında yeni tespit veya yeni doğruluk değerlendirmesi yapılmadı.

Asıl top sınırı devam ediyor: küçük topa dair yeterli YOLO adayı yoksa takipçi doğru gözlem üretemez. Yuvarlak ve hareketli beyaz nesne olmak tek başına güvenilir bir top sınıflandırıcısı yapılmadı; forma, ayakkabı ve çizgiler de bu özellikleri taşıyabilir. Yeni model eğitilmedi; ayrı çok-kareli top modeli bu iterasyonun sonucu değildir.

### Neden bölgesel arama varsayılan kapalı?

İsteğe bağlı deney 268 ek çıkarım geçişi kullandı. NMS sonrasında 379 tam-kare ve 21 tile adayı kaldı; ROI kaynaklı aday kalmadı. Top kabulü yine 54 oldu. Bu klipte ek maliyet için net bir kabul/ROI geri çağırım kazancı kanıtlanmadı. Bu nedenle maliyetli deney otomatik açık bırakılmadı. Başka kamera için `--ball-search` ancak ayrı örneklerle ölçülerek denenmeli. Ara denemeler `results/v031_short/` ve `results/v031_short_refined/` altında; tavsiye edilen son çıktı `results/v031_final/` dizinidir.

## Çıktılar ve tekrar kullanım

- `results/v031_final/tracked.mp4`: son kişi + top inceleme videosu; 450 kare baştan sona çözümlendi.
- `results/v031_final/review_comparison.jpg`: 183, 331, 333 ve 350. karelerde eski/yeni yan yana inceleme.
- `raw_detections.csv` / `raw_detections.meta.json`: başka takip ayarlarına tekrar verilecek ham kayıtlar ve boş kare/kamera bilgisi.
- `player_diagnostics.csv`: etiket kaybının tespit, saha filtresi, eşleştirme veya atama aşamasında incelenmesini sağlar.
- `track_states.csv`: kişi gözlem/atanmamış/tahmin durumları ile top tahminleri. Gözlenen top durumları `ball_track.csv` ve kutuları `detections.csv` içindedir.
- `results/v031_final_csv/`: videosuz CSV tekrar oynatma; model ağırlığı yüklenmeden oluşturuldu.
- `results/v031_final_csv_render/`: aynı CSV'den, yalnızca render için ham video kullanılarak oluşturulan kişi videosu; 450 kareyle doğrulandı.
- `results/v031_replay_strict/`: aynı ham kişi verisiyle 0,25 tabanlı kontrol.
- `results/v031_audit/ball_control.json`, `marking_regression.json`: kontrollü top filtresi ve iki bilinen hata karesinin kayıtları.

```powershell
python run_tracking.py --video ilk_15_saniye.mp4 --output results/new_run
python track_csv.py --detections results/new_run/raw_detections.csv --output results/csv_only
python track_csv.py --detections results/new_run/raw_detections.csv --output results/strict_control --person-floor 0.25 --hold-frames 0
# Kullanıcının verdiği özgün 11 sütunlu dosya, video olmadan:
python track_csv.py --detections provided.csv --fps 30 --frame-count 450 --output results/provided_run
# Kalibrasyonla videosuz saha filtresi (kendi ölçülmüş JSON'unuz):
python track_csv.py --detections provided.csv --fps 30 --frame-count 450 --image-size 640 360 --camera-config my_camera.json --output results/calibrated_replay
```

Eski tracked CSV'den zaten atılmış tespitler geri getirilemez. Ham dosyadaki en düşük güvenin altını sonradan istemek de eksik veriyi oluşturmaz. Parquet henüz desteklenmiyor. CSV-only kişi hattı topun görüntü/arka plan tabanlı kararlarını yeniden üretmez; top hattı ayrı tutuluyor.

## Testler ve tamamlanmayan işler

**31 test geçti.** Önceki 17 teste ek olarak CSV uyumluluğu/NaN reddi, videosuz tekrar oynatma, zayıf gözlemle ID korunması, atanmamış kişinin görünürlüğü, tahmin süresi, ayak poligonu, boyuttan saha dörtgeni, tutarlı yeni top hipotezi, kutu içinde seyrelmiş beyaz nokta, tile/NMS ve alternatif ROI davranışı test edildi. Sentetik projeksiyon testleri gerçek kamera kalibrasyonu doğrulaması değildir. Python derleme/CLI ve çıktı bütünlüğü kontrolleri yapıldı.

Değişen mevcut dosyalar: `run_tracking.py`, `ball_tracking.py`, `pitch_context.py`, `calibration.py`, `configs/camera.example.json`, `README.md`, `AGENTS.md`. Eklenenler: `detection_io.py`, `detector.py`, `player_tracking.py`, `track_csv.py`, `ball_detection.py`, `annotations.py`, `test_detection_tracking.py`, `constraints-tested.txt`, bu rapor. Önceki baseline ve `tracking_helpers.py` korunuyor. Bağımlılık kurulumu/güncellemesi yapılmadı; isteğe bağlı constraints dosyası kurulu ortamın bildirilen kesin sürümlerini kaydeder, tam transitif lock değildir.

Gerçek kalibrasyon/ölçülü yürüme doğrulaması, metre tabanlı takip ve hız/mesafe/ısı haritası, ileri/geri tracklet birleştirme, takım/kaleci/numara sınıflandırması, IDF1/HOTA değerlendirmesi ve Parquet sonraki aşamalardır. Proje LICENSE seçimi sahibiyle kararlaştırılmadı; lisans atanmadı. Bağımlılığı adaptörün arkasına koymak lisans kararının yerine geçmez. Yeni model eğitimi veya etiketli doğrulama kümesi olmadan büyük doğruluk iddiası yoktur.

# v0.3 doğrulama raporu

Bu rapor v0.2 checkpoint'i `da4c260` üzerine, aynı `feature/tracking-v02` dalında yapılan v0.3 geliştirmesini açıklar. v0.2 commit'i `origin` üzerindeki aynı dala gönderildi. Yeni dal açılmadı. Aşağıdaki sonuçlar genel doğruluk garantisi değildir.

## Değişen davranış

- Uzun video tek bir sabit kameraya ait değil; farklı saha ve açılardan kesitler içeriyor. Kesitler otomatik ayrılıyor; her kesit için ayrı medyan arka plan ve beyaz referans maskesi hazırlanıyor. Oyuncu/top takip durumu kesitte sıfırlanıyor.
- Bir top adayı referansta beyaz olan bölgede bulunuyor ve referanstan yeterince farklı görünmüyorsa `static_marking` kararı alıyor. Önceden onaylanmış adaylar da bu kontrolden geçiyor. Eski filtrenin çevre halkasına bakarak kaçırdığı izole beyaz noktalar böylece değerlendirilebiliyor.
- Adayın sabit durması tek başına ret sebebi değil. Referanstan ayırt edilebilen gerçek bir top dururken de kabul edilebiliyor. Buna karşılık, referansın içine yerleşmiş veya beyaz işaretle görsel olarak aynı olan bir top belirsiz kalabilir.
- Yeni top takibi, güven skoru, değişen piksel oranı ve hareket tahminiyle ilişkilendirme yapıyor. Her işlenen karede en fazla bir gözlenen top seçiliyor. Kayıplar için konum uydurulmuyor; görüntüde olmayan karelerde koordinatlar boş bırakılıyor.
- İsteğe bağlı kamera kalibrasyonu için sahneye özel saha poligonu ve görüntü–metre eşleşmeleri okunabiliyor. Gerçek ölçü verilmediğinden bu çalışmalarda metre koordinatı üretilmedi. Dört noktayla düşük uyum hatası elde etmek bağımsız kalibrasyon doğrulaması değildir. Havalanmış topun 3B konumu düzlemsel homografiden çıkarılmıyor.

## Kısa video: aynı koşullarda karşılaştırma

Girdi: `ilk_15_saniye.mp4`; 450 kare, 640×360 piksel, yaklaşık 29,97 FPS. Yerel model `yolo11n.pt`, `imgsz=960`, `stride=1`, kişi güven eşiği 0,25, top güven eşiği 0,15. Yeni çalışmada `threads=1` kullanıldı. Yeni model eğitilmedi; mevcut bağımlılık gereksinimleri değiştirilmedi.

| Ölçüm | v0.2 | v0.3 |
| --- | ---: | ---: |
| Ham YOLO top adayı | 396 | 396 |
| Kabul edilen top kutusu | 375 | 52 |
| En az bir kabul edilmiş top kutusu bulunan kare | 325 | 52 |
| Kişi tespit satırı | 6.146 | 6.146 |
| Farklı, negatif olmayan kişi takip ID'si | 80 | 80 |
| Olası uzamsal ID geçişi | 18 | 18 |
| Çözümlenen çıktı video karesi | 450 | 450 |

Top sayısı karşılaştırması, v0.3'ün kaydettiği aynı YOLO adaylarının değiştirilmemiş `TemporalBall` v0.2 filtresinden yeniden geçirilmesiyle de doğrulandı: 375 kabul. Oyuncu sayıları ayrıca önceki tam v0.2 çalışmasının kayıtlarıyla eşleşiyor. Bu tekrar oynatma aracı eski oyuncu takip hattını yeniden çalıştırmaz.

### Aday kararları ve top durumları

| Aday kararı | Sayı | Anlam |
| --- | ---: | --- |
| `static_marking` | 334 | Beyaz arka planla eşleşen, yeterli görüntü farkı olmayan aday |
| `accepted` | 52 | Seçilmiş ve zamansal olarak onaylanmış gözlem |
| `temporal_unconfirmed` | 4 | Yeterli tekrar bekleyen seçilmiş aday |
| `motion_gate` | 4 | Etkin hipotezin hareket kapısının dışında kalan aday |
| `alternative` | 2 | Aynı karede başka bir aday seçildiği için kullanılmayan aday |

| Kare durumu | Sayı |
| --- | ---: |
| `observed` | 52 |
| `tentative` | 4 |
| `occluded` | 39 |
| `missing` | 355 |

`missing` sayısı gerçek topun kaçırıldığı kesin kare sayısı değildir; tüm kareler için top görünürlüğü etiketi yok. `observed` da doğru top garantisi değildir; kabul edilen aday anlamına gelir. Kısa kayıp durumunda `occluded` ID tutulabilir ama konum ve güven alanları boş kalır.

### İncelenen sabit beyaz bölge

İlk görüntüde yaklaşık `(173,133)` merkezindeki beyaz saha işareti için `[165,126,180,139]` piksel dikdörtgeni incelendi. Bu bölgedeki merkezlere sahip v0.2 kabulü **322**, v0.3 kabulü **0**. Bu, seçilen bölgeye ilişkin bir inceleme ölçümüdür; bütün videoya veya tüm kameralara ait yanlış pozitif oranı değildir. Topun başka bir anda aynı bölgeye gelmesini yasaklayan sabit bir koordinat kuralı algoritmaya eklenmedi. Dikdörtgen yalnızca karşılaştırma aracına verilen inceleme parametresidir.

Örneğin 100. karede adayın YOLO güveni **0,6353** olmasına rağmen arka plandaki beyaz piksel oranı **0,4048**, değişen piksel oranı **0** olduğu için aday elendi. Yüksek model güveninin tek başına doğru top anlamına gelmediği burada görülebiliyor.

### Görsel inceleme ve kalan hatalar

- 202. karede sağ alttaki gerçek top kabul edildi: güven yaklaşık 0,1616, değişen piksel oranı 0,50. Daha yüksek güvenli sabit işaret adayı elendi.
- 350. karede oyuncuların yanındaki gerçek top kabul edildi: güven yaklaşık 0,6421, hareket tahmini farkı yaklaşık 2,76 piksel.
- **331. karede hata devam ediyor:** beyaz formalı oyuncunun alt bölgesindeki aday top olarak seçiliyor; görünen gerçek top doğru seçilmiyor. Hareket kapısı ve önceden seçilmiş yanlış hipotez de doğru adayı dışarıda bırakabilir. Bu kare, beyaz saha işareti filtresinin tek başına oyuncu kıyafeti/ayakkabısı kaynaklı hataları çözmediğini gösteriyor.

Bu üç kare örnek incelemedir; rastgele seçilmiş bağımsız bir test seti veya precision/recall hesabı değildir. İnceleme görselleri: `results/v03_short/review_frames.jpg` ve `marking_review.jpg`.

## Tekrarlama komutları

Her çalıştırmada yeni çıktı dizini seçin; mevcut dizin tekrar kullanılırsa dosyalar üzerine yazılır.

```powershell
python -m unittest -v test_tracking_helpers test_pitch_tracking
python run_tracking.py --video ilk_15_saniye.mp4 --output results/v03_short
python compare_ball_filters.py --video ilk_15_saniye.mp4 --output results/v03_short --review-region 165 126 180 139
python run_tracking.py --video halisaha.mp4 --output results/v03_long_sampled --stride 3 --imgsz 640
python compare_ball_filters.py --video halisaha.mp4 --output results/v03_long_sampled --review-region 165 126 180 139 --review-scene 0
```

## Sınırlar ve sonraki geliştirme

Arka plan örneklemesi çevrimdışıdır ve aynı kesitin sonraki karelerini kullanır; canlı video başarımı iddiası yoktur. Uzun süre aynı yerde duran top referansa karışabilir. Işık değişimi, gölgeler, sıkıştırma, hareketli kamera ve yumuşak sahne geçişleri yöntemi zorlaştırır. Farklı kesitlerin ID'leri aynı kişiyi ifade etmez; ID sayısı gerçek oyuncu sayısı değildir. Kişi sınıfında hakem veya seyirci de bulunabilir.

Sonraki öncelik, küçük ve zor topları kapsayan etiketli bir doğrulama kümesi oluşturmak; durağan top, penaltı noktası, forma/ayakkabı, hızlı top ve örtüşme örneklerini ayrı değerlendirmektir. Ardından yakın bölgeyi daha yüksek çözünürlükte yeniden tarama ve daha sağlam görünüm/hareket ilişkilendirmesi kontrollü olarak karşılaştırılabilir. Gerçek saha ölçüleriyle kalibrasyon doğrulanmadan hız ve mesafe hesaplanmamalı. Eğitim ancak uygun etiketli eğitim verisi varsa yapılmalı.

## Do?rulama ve dosyalar

17 test ge?ti: ?nceki 5 yard?mc? test, 10 top/saha/kalibrasyon testi ve 2 kesit s?n?r?/?zerine yazmama testi. Python derleme ve CLI kontrolleri ge?ti. K?sa ??kt? 450, uzun ?rneklemeli ??kt? 1.463 kareyle eksiksiz ??z?mlendi. Ham kesitler 4.388, a??klamal? kesitler 1.463 kareye toplamland?. Mevcut model, gereksinimler, v0.1 baseline ve v0.2 `tracking_helpers.py` de?i?tirilmedi.

De?i?en mevcut dosyalar: `run_tracking.py`, `README.md`, `AGENTS.md`. Eklenen mod?ller: `ball_tracking.py`, `pitch_context.py`, `calibration.py`, `compare_ball_filters.py`, `split_scenes.py`; yeni testler: `test_pitch_tracking.py`, `test_split_scenes.py`; yap?land?rma ?ablonu: `configs/camera.example.json`; bu rapor: `reports/v03_validation.md`. Video/CSV ??kt?lar? git taraf?ndan yok say?lan `results/` alt?nda. v0.3 de?i?iklikleri inceleme i?in ?al??ma a?ac?nda b?rak?ld?; bu rapor haz?rlan?rken v0.3 commit/push yap?lmad?.

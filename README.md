# Gerçek Zamanlı Obje Tespiti ve Trafik Kural İhlali Analizi

Bu proje, bilgisayarla görü teknikleri kullanılarak yaya ve taşıtları gerçek zamanlı olarak tespit etmeyi ve trafik kurallarına uyumu analiz etmeyi amaçlamaktadır. Sistem, video akışı üzerinden gelen görüntülerde nesneleri tanır, konumlarını analiz eder ve belirli bölgeler ile çakışmaları değerlendirerek ihlalleri işaretler.

---

## Projenin Amacı

- Dörtyol ve yaya geçidi gibi kritik bölgelerde yaya ve taşıt tespiti
- Trafik kurallarına aykırı hareketlerin otomatik olarak belirlenmesi
- İhlal durumlarının kullanıcıya görsel olarak gösterilmesi

---

## Kullanılan Teknolojiler

- **Python 3**
- `OpenCV` – Görüntü işleme
- `NumPy` – Matematiksel işlemler
- `cv2.createBackgroundSubtractorMOG2()` – Arka plan çıkarımı
- `cv2.findContours()` – Kontur analizi
- `cv2.pointPolygonTest()` – Bölge çakışma kontrolü

---

## Algoritmanın Akışı

1. Video kaynağı alınır
2.  Her kare için:
   - Grayscale dönüşüm + Gaussian Blur uygulanır
   - Arka plan çıkarımı yapılır (MOG2)
   - Eşikleme ve morfolojik işlemler ile görüntü netleştirilir
   - Kontur analizi ile nesneler tespit edilir
   - Boy/en oranına göre **yaya** veya **taşıt** olarak sınıflandırılır
   - Tanımlı bölgelere (dörtyol/yaya geçidi) göre **ihlal analizi** yapılır
   - Her nesne, sınıfına göre renkli olarak çerçevelenir
   - Görüntü üzerine ihlal etiketi eklenir

---

##  Tanımlı Bölgeler

-  **Dörtyol Bölgesi:** Yayaların geçmemesi gereken alan
-  **Yaya Geçidi:** Taşıtların ihlal etmemesi gereken alan

---

## Sonuçlar

- Yüksek doğrulukla yaya ve taşıt tespiti sağlanmıştır.
- Tanımlı alanlara göre ihlaller başarılı şekilde algılanmıştır.
- Gerçek zamanlı görsel uyarı mekanizması çalışmaktadır.

---

## Projeyi Çalıştırmak İçin

### Gerekli Kütüphaneleri Yükleyin:

```bash
pip install opencv-python numpy
```

### Ardından `proje_odevi.py` dosyasını çalıştırın:

```bash
python proje_odevi.py
```

> **Not:** `Train.mp4` ve `Test.mp4` dosyalarının proje dizininde bulunduğundan emin olun.

---

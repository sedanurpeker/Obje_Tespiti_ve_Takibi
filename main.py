import cv2
import numpy as np

# Video kaynaklarını aç
train_video = cv2.VideoCapture(r"C:\Users\user\PycharmProjects\goruntu_isleme\Train.mp4")
test_video = cv2.VideoCapture(r"C:\Users\user\PycharmProjects\goruntu_isleme\Test.mp4")

# Arka plan çıkarıcıyı oluştur
fgbg = cv2.createBackgroundSubtractorMOG2()


# Nesne izleme yapısı
class ObjectTracker:
    def __init__(self):
        self.objects = {}
        self.next_object_id = 0

    def update(self, detected_objects):
        updated_objects = {}
        for obj in detected_objects:
            key = (obj['koordinatlar'][0], obj['koordinatlar'][1])
            if key in self.objects:
                prev_obj = self.objects[key]
                obj['hiz'] = np.linalg.norm(
                    np.array(obj['koordinatlar'][:2]) - np.array(prev_obj['koordinatlar'][:2])
                )  # Hız hesapla (frame başına hareket)
                obj['id'] = prev_obj['id']
            else:
                obj['hiz'] = 0  # Yeni obje için hız başlangıçta 0
                obj['id'] = self.next_object_id
                self.next_object_id += 1
            updated_objects[key] = obj
        self.objects = updated_objects
        return list(updated_objects.values())


# Nesne izleyiciyi başlat
tracker = ObjectTracker()


def tespit_ve_siniflandir(cerceve, arka_plan_cikarici, tracker):
    # Arka plan çıkarma
    fgmask = arka_plan_cikarici.apply(cerceve)

    # Kenar tespiti ve kontur bulma
    kenarlar = cv2.Canny(fgmask, 50, 150)
    konturlar, _ = cv2.findContours(kenarlar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    objeler = []
    # Bulunan konturlar içinde nesneleri tespit etme
    for kontur in konturlar:
        if cv2.contourArea(kontur) < 500:  # Küçük konturları filtrele
            continue

        # Kontur çevresine dikdörtgen çiz
        (x, y, w, h) = cv2.boundingRect(kontur)

        # Sınıflandırma (örnek, taşıt boyutları geniş kabul ediliyor)
        if w < 40 and h < 70:
            tip = 'yaya'
        elif 40 <= w < 100:  # Bisiklet veya küçük taşıtlar
            tip = 'tasit'
        else:  # Daha büyük taşıtlar
            tip = 'tasit'

        objeler.append({'tip': tip, 'koordinatlar': (x, y, w, h), 'hiz': 0})

    # Nesneleri izleyici ile güncelle
    objeler = tracker.update(objeler)
    return objeler


def isleme(video):
    # Sayaçları sıfırla
    global yaya_sayaci, tasit_sayaci, tracker
    yaya_sayaci = set()
    tasit_sayaci = set()
    tracker = ObjectTracker()

    while video.isOpened():
        ret, cerceve = video.read()
        if not ret:
            break

        objeler = tespit_ve_siniflandir(cerceve, fgbg, tracker)

        for obje in objeler:
            x, y, w, h = obje['koordinatlar']
            if obje['tip'] == 'yaya':
                yaya_sayaci.add(obje['id'])
                cv2.rectangle(cerceve, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Yeşil
            else:
                tasit_sayaci.add(obje['id'])
                cv2.rectangle(cerceve, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Kırmızı

            # Nesne bilgilerini göster
            cv2.putText(
                cerceve,
                f"{obje['tip']}, {obje['hiz']:.2f} px/frame",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Toplam yaya ve taşıt sayısını göster
        cv2.putText(
            cerceve,
            f"Yaya: {len(yaya_sayaci)} Tasit: {len(tasit_sayaci)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Videoyu göster
        cv2.imshow('Nesne Tespiti', cerceve)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


# Eğitim videosunu işleme
isleme(train_video)
# Test videosunu işleme
isleme(test_video)

# Sonuçları yazdır
print(f"Toplam Yaya Sayısı: {len(yaya_sayaci)}")
print(f"Toplam Taşıt Sayısı: {len(tasit_sayaci)}")

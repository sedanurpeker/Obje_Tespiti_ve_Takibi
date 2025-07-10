import cv2
import numpy as np

# Video kaynaklarını aç
train_video = cv2.VideoCapture(r"C:\Users\user\PycharmProjects\goruntu_isleme\Train.mp4")
test_video = cv2.VideoCapture(r"C:\Users\user\PycharmProjects\goruntu_isleme\Test.mp4")

# Bölgelerin tanımlanması
dort_yol_bolgesi = [(50, 50), (300, 50), (300, 300), (50, 300)]
yaya_gecidi_bolgesi = [(200, 250), (350, 250), (350, 300), (200, 300)]

# Arka plan çıkarıcıyı oluştur
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

def bolge_kontrolu(nesne_orta, bolge):
    x, y = nesne_orta
    return cv2.pointPolygonTest(np.array(bolge, dtype=np.int32), (x, y), False) >= 0

def isleme(video, fgbg):
    while video.isOpened():
        ret, cerceve = video.read()
        if not ret:
            break

        # Grayscale dönüşümü ve blurlama
        gri_cerceve = cv2.cvtColor(cerceve, cv2.COLOR_BGR2GRAY)
        gri_cerceve = cv2.GaussianBlur(gri_cerceve, (5, 5), 0)

        # Arka plan çıkarma
        fgmask = fgbg.apply(gri_cerceve)
        _, esiklenmis = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Morfolojik işlemler
        kernel = np.ones((3, 3), np.uint8)
        esiklenmis = cv2.erode(esiklenmis, kernel, iterations=1)
        esiklenmis = cv2.dilate(esiklenmis, kernel, iterations=2)

        # Kontur bulma
        konturlar, _ = cv2.findContours(esiklenmis, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for kontur in konturlar:
            if cv2.contourArea(kontur) < 500:
                continue

            # Nesne koordinatları
            x, y, w, h = cv2.boundingRect(kontur)
            nesne_orta = (x + w // 2, y + h // 2)

            # Nesnenin türünü tahmin et (yaya veya taşıt)
            if h > w * 1.5:
                nesne_tipi = "yaya"
            else:
                nesne_tipi = "tasit"

            # Bölge ihlali kontrolü
            if nesne_tipi == "yaya" and not bolge_kontrolu(nesne_orta, dort_yol_bolgesi):
                # Yaya dörtyol kurallarını ihlal ediyor
                cv2.rectangle(cerceve, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(cerceve, "Yaya Ihlali", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            elif nesne_tipi == "tasit" and bolge_kontrolu(nesne_orta, yaya_gecidi_bolgesi):
                # Taşıt yaya geçidinde ve durmamış
                cv2.rectangle(cerceve, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(cerceve, "Tasit Ihlali", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # Nesne kurallara uygun hareket ediyor
                renk = (0, 255, 0) if nesne_tipi == "yaya" else (255, 255, 0)
                cv2.rectangle(cerceve, (x, y), (x + w, y + h), renk, 2)

        # Görselleştirme: Bölgeleri çiz
        cv2.polylines(cerceve, [np.array(dort_yol_bolgesi, dtype=np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(cerceve, [np.array(yaya_gecidi_bolgesi, dtype=np.int32)], True, (255, 0, 0), 2)

        # Görüntüyü göster
        cv2.imshow("Video", cerceve)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Eğitim videosunu işleme
isleme(train_video, fgbg)
# Test videosunu işleme
isleme(test_video, fgbg)

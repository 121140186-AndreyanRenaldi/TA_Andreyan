import cv2

# Path ke video Anda
video_path = 'TENGAH-9.mov'  # Ganti dengan path video Anda

# Buka video
cap = cv2.VideoCapture(video_path)

# Ambil frame pertama
success, frame = cap.read()

# Simpan jika berhasil
if success:
    cv2.imwrite('tengah_9.jpg', frame)
    print("Berhasil menyimpan frame pertama.")
else:
    print("Gagal membaca frame pertama.")

# Tutup video
cap.release()
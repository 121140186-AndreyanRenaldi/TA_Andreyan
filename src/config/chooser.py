from config.scheme import SCHEMES, SCHEME_ORDER

def choose_scheme():
    print("=== Pilih Skema Data ===")
    for key, name in SCHEME_ORDER.items():
        print(f"{key}. {name}")
    print("========================")

    while True:
        try:
            idx = int(input("Masukkan angka (1-4): "))
            if idx in SCHEME_ORDER:
                scheme_key = SCHEME_ORDER[idx]
                return SCHEMES[scheme_key]
            else:
                print("Pilihan harus 1-4. Coba lagi.")
        except ValueError:
            print("Input tidak valid. Masukkan angka 1-4.")

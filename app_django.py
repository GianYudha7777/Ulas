import sys
import joblib
from django.conf import settings
from django.core.management import execute_from_command_line
from django.http import HttpResponse
from django.urls import path
from django.utils.html import escape
from django.views.decorators.csrf import csrf_exempt

settings.configure(
    DEBUG=True,
    SECRET_KEY="ulas-liquid-glass-key",
    ROOT_URLCONF=__name__,
    ALLOWED_HOSTS=["*"],
)

vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model_sentimen.joblib")


def Analisis_sentimen(teks):
    fitur = vectorizer.transform([teks])
    hasil = model.predict(fitur)[0]
    keyakinan = model.predict_proba(fitur).max() * 100
    return hasil, keyakinan

@csrf_exempt
def halaman_utama(request):
    ulasan = ""
    tab_aktif = "beranda"
    blok_hasil = ""

    if request.method == "POST":
        tab_aktif = "Analisis"
        ulasan = request.POST.get("ulasan", "").strip()
        if ulasan:
            hasil, keyakinan = Analisis_sentimen(ulasan)
            kelas = "pos" if hasil == "Positif" else "neg"
            blok_hasil = f"""
            <div class="result {kelas}">
                <span>Sentimen: <b>{hasil.upper()}</b></span>
                <span class="badge">Keyakinan {keyakinan:.1f}%</span>
            </div>
            """

    html = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ulas — Sistem Analisis Sentimen</title>
        <style>
            :root {{
                --bg: #eef1f5;
                --glass: rgba(255, 255, 255, 0.58);
                --glass-strong: rgba(255, 255, 255, 0.85);
                --border: rgba(255, 255, 255, 0.9);
                --line: rgba(15, 23, 42, 0.08);
                --text: #111827;
                --muted: #6b7280;
                --accent: #18181b;
            }}
            * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }}
            body {{
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                color: var(--text);
                min-height: 100vh;
                position: relative;
                overflow-x: hidden;
            }}
            .orb {{
                position: fixed;
                border-radius: 50%;
                filter: blur(85px);
                z-index: -1;
            }}
            .orb-1 {{ width: 420px; height: 420px; background: #cbd5e1; top: -120px; left: -100px; opacity: 0.7; }}
            .orb-2 {{ width: 380px; height: 380px; background: #ffffff; bottom: -100px; right: -80px; opacity: 0.9; }}
            .container {{ max-width: 920px; margin: 0 auto; padding: 28px 24px 60px; }}
            header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding-bottom: 18px;
                border-bottom: 1px solid var(--line);
            }}
            .brand {{ font-size: 22px; font-weight: 700; letter-spacing: -0.5px; display: flex; align-items: center; gap: 9px; }}
            .brand-dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--accent); }}
            .subtitle-top {{ font-size: 11px; letter-spacing: 1.8px; color: var(--muted); text-transform: uppercase; }}
            nav {{ display: flex; gap: 8px; margin: 22px 0 30px; flex-wrap: wrap; }}
            .nav-btn {{
                background: transparent;
                color: var(--muted);
                border: 1px solid transparent;
                padding: 8px 16px;
                border-radius: 12px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: 0.2s;
            }}
            .nav-btn:hover {{ color: var(--text); background: rgba(255, 255, 255, 0.4); }}
            .nav-btn.active {{
                color: var(--text);
                background: var(--glass-strong);
                border-color: var(--border);
                backdrop-filter: blur(16px);
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            }}
            .glass-card {{
                background: var(--glass);
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 34px;
                backdrop-filter: blur(22px);
                -webkit-backdrop-filter: blur(22px);
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06), inset 0 1px 0 #ffffff;
            }}
            .section {{ display: none; animation: fadeIn 0.25s ease; }}
            .section.active {{ display: block; }}
            @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(5px); }} to {{ opacity: 1; transform: translateY(0); }} }}
            .tag {{ color: var(--muted); font-size: 12px; letter-spacing: 1.6px; text-transform: uppercase; margin-bottom: 12px; display: block; }}
            h1 {{ font-size: 38px; line-height: 1.15; margin-bottom: 16px; letter-spacing: -0.8px; }}
            h2 {{ font-size: 24px; margin-bottom: 12px; letter-spacing: -0.4px; }}
            h3 {{ font-size: 16px; margin-bottom: 8px; color: var(--text); }}
            p {{ color: var(--muted); line-height: 1.65; margin-bottom: 18px; }}
            .btn-group {{ display: flex; gap: 12px; margin-top: 24px; }}
            .btn-primary {{
                background: var(--accent);
                color: #ffffff;
                border: none;
                padding: 12px 22px;
                border-radius: 12px;
                font-weight: 600;
                font-size: 14px;
                cursor: pointer;
                transition: 0.2s;
            }}
            .btn-primary:hover {{ opacity: 0.88; transform: translateY(-1px); }}
            .btn-ghost {{
                background: rgba(255, 255, 255, 0.55);
                color: var(--text);
                border: 1px solid var(--border);
                padding: 12px 22px;
                border-radius: 12px;
                font-size: 14px;
                cursor: pointer;
                backdrop-filter: blur(12px);
            }}
            textarea {{
                width: 100%;
                height: 120px;
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(15, 23, 42, 0.12);
                border-radius: 14px;
                padding: 14px;
                color: var(--text);
                font-size: 15px;
                resize: none;
                outline: none;
                margin-bottom: 16px;
            }}
            textarea:focus {{ border-color: var(--accent); background: #ffffff; }}
            .result {{
                margin-top: 20px;
                padding: 16px 20px;
                border-radius: 14px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                backdrop-filter: blur(12px);
            }}
            .result.pos {{ background: rgba(22, 163, 74, 0.08); border: 1px solid rgba(22, 163, 74, 0.25); color: #15803d; }}
            .result.neg {{ background: rgba(220, 38, 38, 0.08); border: 1px solid rgba(220, 38, 38, 0.25); color: #b91c1c; }}
            .badge {{ background: rgba(255, 255, 255, 0.75); padding: 4px 12px; border-radius: 99px; font-size: 13px; color: var(--text); }}
            .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 18px; }}
            .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 20px; }}
            .info-box {{
                background: rgba(255, 255, 255, 0.55);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 20px;
            }}
            .info-box p {{ margin-bottom: 0; font-size: 14px; }}
            .info-box ul {{ padding-left: 18px; color: var(--muted); font-size: 14px; line-height: 1.65; }}
            .metric-val {{ font-size: 30px; font-weight: 700; color: var(--text); margin: 6px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 18px; font-size: 14px; background: rgba(255,255,255,0.45); border-radius: 12px; overflow: hidden; }}
            th, td {{ border: 1px solid var(--line); padding: 12px; text-align: center; }}
            th {{ background: rgba(255, 255, 255, 0.7); color: var(--text); }}
            td {{ color: var(--muted); }}
        </style>
    </head>
    <body>
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="container">
            <header>
                <div class="brand"><span class="brand-dot"></span> ulas.</div>
                <div class="subtitle-top">Sistem Analisis Sentimen</div>
            </header>

            <nav>
                <button class="nav-btn" data-tab="beranda" onclick="bukaTab('beranda')">Beranda</button>
                <button class="nav-btn" data-tab="Analisis" onclick="bukaTab('Analisis')">Ruang Analisis</button>
                <button class="nav-btn" data-tab="evaluasi" onclick="bukaTab('evaluasi')">Evaluasi Model</button>
                <button class="nav-btn" data-tab="panduan" onclick="bukaTab('panduan')">Panduan</button>
            </nav>

            <div id="beranda" class="section">
                <div class="glass-card">
                    <span class="tag">Klasifikasi Teks Bahasa Indonesia</span>
                    <h1>Pahami ulasan.<br>Deteksi sentimen seketika.</h1>
                    <p>Ulas membantu memilah ulasan pengguna berbahasa Indonesia menjadi kategori <b>Positif</b> atau <b>Negatif</b> secara otomatis menggunakan ekstraksi fitur Bag of Words dan algoritma Multinomial Naive Bayes.</p>
                    <div class="btn-group">
                        <button class="btn-primary" onclick="bukaTab('Analisis')">Mulai Analisis ↗</button>
                        <button class="btn-ghost" onclick="bukaTab('panduan')">Pelajari Cara Kerja</button>
                    </div>
                </div>
            </div>

            <div id="Analisis" class="section">
                <div class="glass-card">
                    <span class="tag">Ruang Analisis</span>
                    <h2>Analisis Sentimen Ulasan</h2>
                    <p>Ketik atau tempel kalimat ulasan produk maupun aplikasi di bawah ini.</p>
                    <form method="post">
                        <textarea name="ulasan" placeholder="Contoh: Aplikasinya sangat membantu dan prosesnya cepat sekali">{escape(ulasan)}</textarea>
                        <button type="submit" class="btn-primary">Analisis Sentimen</button>
                    </form>
                    {blok_hasil}
                </div>
            </div>

            <div id="evaluasi" class="section">
                <div class="glass-card">
                    <span class="tag">Evaluasi Model</span>
                    <h2>Performa & Detail Metode</h2>
                    <p>Model dilatih menggunakan 1.595 ulasan bersih (75% data latih, 25% data uji).</p>

                    <div class="grid-3">
                        <div class="info-box"><div class="subtitle-top">Akurasi Uji</div><div class="metric-val">84.0%</div><p style="font-size:13px;">Data uji terpisah (399 ulasan)</p></div>
                        <div class="info-box"><div class="subtitle-top">Akurasi Latih</div><div class="metric-val">88.9%</div><p style="font-size:13px;">Data latih (1.196 ulasan)</p></div>
                        <div class="info-box"><div class="subtitle-top">Total Data Bersih</div><div class="metric-val">1.595</div><p style="font-size:13px;">854 Negatif · 741 Positif</p></div>
                    </div>

                    <table>
                        <tr><th>Aktual / Prediksi</th><th>Prediksi Positif</th><th>Prediksi Negatif</th></tr>
                        <tr><td><b>Aktual Positif</b></td><td>134</td><td>51</td></tr>
                        <tr><td><b>Aktual Negatif</b></td><td>13</td><td>201</td></tr>
                    </table>

                    <div class="grid-2">
                        <div class="info-box">
                            <h3>Metode yang Digunakan</h3>
                            <p><b>Multinomial Naive Bayes</b> yang dipadukan dengan ekstraksi fitur teks <b>CountVectorizer (Bag of Words)</b> untuk klasifikasi teks biner (Positif & Negatif).</p>
                        </div>
                        <div class="info-box">
                            <h3>Cara Kerja Metode</h3>
                            <p>Teks ulasan dipecah menjadi hitungan frekuensi kata. Model lalu menghitung peluang kemunculan setiap kata pada kelas Positif dan Negatif berdasarkan Teorema Bayes, kemudian memilih kelas dengan probabilitas tertinggi.</p>
                        </div>
                        <div class="info-box">
                            <h3>Kelebihan</h3>
                            <ul>
                                <li>Proses pelatihan dan Analisis sangat cepat serta ringan.</li>
                                <li>Bekerja baik pada data teks berdimensi tinggi meski ukuran dataset relatif kecil.</li>
                                <li>Alur model sederhana dan mudah diinterpretasikan.</li>
                            </ul>
                        </div>
                        <div class="info-box">
                            <h3>Kekurangan</h3>
                            <ul>
                                <li>Mengasumsikan setiap kata berdiri sendiri sehingga belum memahami konteks negasi (misal: <i>"kurang bagus"</i>).</li>
                                <li>Sensitif terhadap kata yang jarang muncul di data latih.</li>
                                <li>Hanya mendukung 2 kelas (belum memiliki kategori Netral).</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <div id="panduan" class="section">
                <div class="glass-card">
                    <span class="tag">Panduan Sistem</span>
                    <h2>Alur Kerja & Penggunaan</h2>
                    <p><b>1. Preprocessing:</b> Data kosong, duplikat, dan rating netral (bintang 3) disisihkan. Rating 4–5 dipetakan menjadi Positif dan 1–2 menjadi Negatif.</p>
                    <p><b>2. Pembagian Data:</b> Dataset dibagi 75% untuk pelatihan dan 25% untuk pengujian tersegel agar evaluasi tidak bocor.</p>
                    <p><b>3. Analisis:</b> Buka menu <b>Ruang Analisis</b>, masukkan kalimat ulasan berbahasa Indonesia, lalu tekan tombol analisis.</p>
                </div>
            </div>
        </div>

        <script>
            function bukaTab(nama) {{
                document.querySelectorAll('.section').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
                document.getElementById(nama).classList.add('active');
                document.querySelector('[data-tab="' + nama + '"]').classList.add('active');
            }}
            bukaTab("{tab_aktif}");
        </script>
    </body>
    </html>
    """
    return HttpResponse(html)


urlpatterns = [
    path("", halaman_utama),
]


if __name__ == "__main__":
    execute_from_command_line(sys.argv)
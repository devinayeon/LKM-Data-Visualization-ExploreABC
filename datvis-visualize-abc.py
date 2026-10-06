"""Analisis dan visualisasi kasus NusaKirim dari LKM.
Python >=3.11. Dependensi tercantum pada requirements.txt.
"""

from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parent)
Path("hasil").mkdir(exist_ok=True)
import numpy as np
import pandas as pd
from pathlib import Path
import sys, json

sys.stdout.reconfigure(encoding="utf-8")
rng = np.random.default_rng(42)
gudang = ["A", "B", "C", "D", "E"]
hari = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
shift = ["Pagi", "Siang", "Sore", "Malam"]
params = {"A": (20, 4), "B": (22, 3), "C": (18, 9), "D": (26, 5), "E": (21, 4)}
rows = []
for g in gudang:
    mu, sd = params[g]
    for h in hari:
        for s in shift:
            waktu = rng.normal(mu, sd, 30)
            if g == "E" and h in ["Jum", "Sab"] and s in ["Sore", "Malam"]:
                waktu += 12
            if g == "B" and h == "Sen" and s == "Pagi":
                waktu += 8
            if g == "C":
                waktu += (rng.random(30) < 0.08) * 30
            for w in np.clip(waktu, 4, None):
                rows.append((g, h, s, round(w, 1)))
df = pd.DataFrame(rows, columns=["gudang", "hari", "shift", "jam_kirim"])
df["melanggar_sla"] = df.jam_kirim > 24
scores = pd.DataFrame(
    [
        [78, 85, 80, 70, 82, 75],
        [74, 80, 78, 85, 79, 83],
        [88, 82, 84, 80, 76, 86],
        [55, 95, 88, 60, 90, 80],
        [85, 78, 70, 88, 72, 77],
    ],
    index=gudang,
    columns=[
        "Kecepatan",
        "Akurasi pesanan",
        "Kepuasan",
        "Efisiensi biaya",
        "Rendah retur",
        "Ketersediaan stok",
    ],
)
summary = df.groupby("gudang").agg(
    n=("jam_kirim", "size"),
    mean=("jam_kirim", "mean"),
    median=("jam_kirim", "median"),
    sd=("jam_kirim", "std"),
    violations=("melanggar_sla", "sum"),
    pct=("melanggar_sla", "mean"),
    min=("jam_kirim", "min"),
    max=("jam_kirim", "max"),
)
summary["pct"] *= 100
summary["q1"] = df.groupby("gudang").jam_kirim.quantile(0.25)
summary["q3"] = df.groupby("gudang").jam_kirim.quantile(0.75)
summary["p95"] = df.groupby("gudang").jam_kirim.quantile(0.95)
summary["score"] = scores.mean(axis=1)
print(summary.to_string())
cell = df.groupby(["gudang", "hari", "shift"]).agg(
    n=("jam_kirim", "size"),
    violations=("melanggar_sla", "sum"),
    mean=("jam_kirim", "mean"),
    pct=("melanggar_sla", "mean"),
)
cell["pct"] *= 100
print("\nCELLS\n" + cell.to_string())
epeak = df[
    (df.gudang == "E")
    & df.hari.isin(["Jum", "Sab"])
    & df["shift"].isin(["Sore", "Malam"])
]
eb = df[
    (df.gudang == "E")
    & ~((df.hari.isin(["Jum", "Sab"])) & (df["shift"].isin(["Sore", "Malam"])))
]
bpeak = df[(df.gudang == "B") & (df.hari == "Sen") & (df["shift"] == "Pagi")]
bb = df[(df.gudang == "B") & ~((df.hari == "Sen") & (df["shift"] == "Pagi"))]
for name, x in [("Epeak", epeak), ("Ebase", eb), ("Bpeak", bpeak), ("Bbase", bb)]:
    print(
        name,
        len(x),
        x.melanggar_sla.sum(),
        100 * x.melanggar_sla.mean(),
        x.jam_kirim.mean(),
    )
assert len(df) == 4200 and summary.n.eq(840).all() and cell.n.eq(30).all()
assert df.notna().all().all() and df.jam_kirim.ge(4).all()
for g in gudang:
    x = df[df.gudang == g].jam_kirim.to_numpy()
    assert np.isclose(summary.loc[g, "sd"], np.std(x, ddof=1))
assert (df.melanggar_sla == (df.jam_kirim > 24)).all()
print(
    "CHECK: 4.200 catatan; 840/gudang; 30/sel; tidak ada nilai hilang; simpangan baku ddof=1; SLA >24; valid."
)
Path("hasil/summary.json").write_text(summary.to_json(orient="index"), encoding="utf-8")
Path("hasil/versions.json").write_text(
    json.dumps({"numpy": np.__version__, "pandas": pd.__version__}), encoding="utf-8"
)
df.to_csv("hasil/data_pengiriman.csv", index=False)
summary.to_csv("hasil/statistik.csv")
cell.to_csv("hasil/hari_shift.csv")
scores.to_csv("hasil/skor.csv")

# Visualisasi NusaKirim. Jalankan setelah analysis.py.
from pathlib import Path
import numpy as np
import pandas as pd
import os

os.environ["MPLCONFIGDIR"] = str(Path("hasil/mplconfig").resolve())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

OUT = Path("hasil/visualisasi")
OUT.mkdir(exist_ok=True)
df = pd.read_csv("hasil/data_pengiriman.csv")
scores = pd.read_csv("hasil/skor.csv", index_col=0)
summary = pd.read_csv("hasil/statistik.csv", index_col=0)
gudang = list("ABCDE")
hari = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
shift = ["Pagi", "Siang", "Sore", "Malam"]
COL = {"A": "#0072B2", "B": "#E69F00", "C": "#8D58A6", "D": "#009E9A", "E": "#D55E00"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.dpi": 220,
    }
)
fmt = lambda x, n=1: f"{x:.{n}f}".replace(".", ",")


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_color("#aaaaaa")


def draw_a(fig, rect, font=11):
    x, y, w, h = rect
    axes = []
    labels = [
        "Kecepatan",
        "Akurasi pesanan",
        "Kepuasan",
        "Efisiensi biaya",
        "Rendah retur",
        "Ketersediaan stok",
    ]
    for k, label in enumerate(labels):
        col = k % 3
        row = k // 3
        ax = fig.add_axes(
            [
                x + col * w / 3 + 0.025 * w,
                y + (1 - row) * h / 2 + 0.11 * h,
                w / 3 - 0.09 * w,
                h / 2 - 0.22 * h,
            ]
        )
        axes.append(ax)
        ax.set_title(label, fontsize=font, pad=5, loc="left")
        for i, g in enumerate(gudang):
            v = scores.loc[g, label]
            ax.scatter(v, i, color=COL[g], s=45, zorder=3)
            ax.text(v + 3, i, str(v), va="center", fontsize=font - 1)
        ax.set_yticks(range(5), gudang)
        ax.set_ylim(4.6, -0.6)
        ax.set_xlim(0, 110)
        ax.set_xticks([0, 50, 100])
        ax.tick_params(labelsize=font - 1)
        ax.grid(axis="x", color="#dddddd", linewidth=0.5)
        ax.set_axisbelow(True)
        clean(ax)
        if row == 1:
            ax.set_xlabel("Skor (0–100)", fontsize=font - 1)
    return axes


def draw_b(fig, rect, font=11):
    x, y, w, h = rect
    ax = fig.add_axes([x + 0.025 * w, y + 0.17 * h, w * 0.65, h * 0.78])
    bar = fig.add_axes([x + 0.76 * w, y + 0.17 * h, w * 0.21, h * 0.78])
    bp = ax.boxplot(
        [df[df.gudang == g].jam_kirim for g in gudang],
        orientation="horizontal",
        patch_artist=True,
        positions=np.arange(5),
        widths=0.55,
        whis=1.5,
        showfliers=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "white",
            "markersize": 4,
        },
        medianprops={"color": "black", "linewidth": 1.6},
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "alpha": 0.5,
            "markerfacecolor": "#666666",
            "markeredgecolor": "none",
        },
    )
    for b, g in zip(bp["boxes"], gudang):
        b.set_facecolor(COL[g])
        b.set_alpha(0.65)
    ax.set_yticks(range(5), gudang)
    ax.invert_yaxis()
    ax.set_xlim(0, 75)
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.set_xlabel("Waktu pengiriman (jam)", fontsize=font)
    ax.axvline(24, color="#333333", linestyle="--", linewidth=1.2)
    ax.text(24.7, -0.55, "SLA 24 jam", fontsize=font - 1)
    ax.annotate(
        "C: maksimum 70,4 jam",
        xy=(70.4, 2),
        xytext=(39, 1),
        fontsize=font - 1,
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    clean(ax)
    bar.barh(
        range(5),
        summary.loc[gudang, "pct"],
        color=[COL[g] for g in gudang],
        height=0.55,
    )
    bar.set_yticks(range(5), gudang)
    bar.invert_yaxis()
    bar.set_xlim(0, 85)
    bar.set_xticks([0, 40, 80])
    bar.set_xlabel("Melanggar SLA (%)", fontsize=font)
    for i, g in enumerate(gudang):
        bar.text(
            summary.loc[g, "pct"] + 2,
            i,
            fmt(summary.loc[g, "pct"], 2) + "%",
            va="center",
            fontsize=font - 1,
        )
    for a in [ax, bar]:
        a.tick_params(labelsize=font - 1)
    clean(bar)
    return ax, bar


def draw_c(fig, rect, font=11):
    x, y, w, h = rect
    axes = []
    im = None
    for i, g in enumerate(gudang):
        ax = fig.add_axes(
            [x + i * w / 5 + 0.027 * w, y + 0.17 * h, w / 5 - 0.049 * w, h * 0.69]
        )
        axes.append(ax)
        mat = (
            df[df.gudang == g]
            .groupby(["hari", "shift"])
            .melanggar_sla.mean()
            .unstack()
            .reindex(index=hari, columns=shift)
            * 100
        )
        im = ax.imshow(mat.to_numpy(), vmin=0, vmax=100, cmap="YlOrBr", aspect="auto")
        ax.set_title("Gudang " + g, color=COL[g], fontsize=font, pad=7)
        ax.set_xticks(range(4), shift, rotation=40, ha="right")
        ax.set_yticks(range(7), hari if i == 0 else [""] * 7)
        ax.tick_params(labelsize=font - 1, length=0)
        if i == 0:
            ax.set_ylabel("Hari", fontsize=font)
        for r in range(7):
            for c in range(4):
                v = mat.iloc[r, c]
                ax.text(
                    c,
                    r,
                    fmt(v, 0),
                    ha="center",
                    va="center",
                    color="white" if v >= 65 else "#171717",
                    fontsize=font - 2,
                )
        if g == "B":
            ax.add_patch(
                Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="#111111", lw=1.7)
            )
        if g == "E":
            ax.add_patch(
                Rectangle((1.5, 3.5), 2, 2, fill=False, edgecolor="#111111", lw=1.7)
            )
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([x + 0.16 * w, y - 0.05 * h, w * 0.68, h * 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=font - 1)
    cb.set_label(
        "Pelanggaran SLA (%) | jam_kirim > 24 | n = 30 per sel", fontsize=font - 1
    )
    return axes


# Tiga gambar terpisah.
f = plt.figure(figsize=(12, 7))
f.text(
    0.055,
    0.955,
    "C unggul pada 2 dari 6 dimensi; D pada 3 dimensi",
    fontsize=18,
    weight="bold",
)
f.text(
    0.055,
    0.91,
    "Data 1 | skor 0–100, semakin tinggi semakin baik | posisi dan angka lebih mudah dibandingkan",
    fontsize=11,
)
draw_a(f, [0.04, 0.09, 0.93, 0.78], 13)
f.text(
    0.06,
    0.025,
    "Sumber: LKM NusaKirim. Rata-rata enam skor: A 78,33 | B 79,83 | C 82,67 | D 78,00 | E 78,33.",
    fontsize=11,
)
f.savefig(OUT / "kebutuhan_A.png")
plt.close(f)
f = plt.figure(figsize=(12, 4.9))
f.text(
    0.055,
    0.95,
    "A paling cepat dan paling jarang melanggar SLA",
    fontsize=18,
    weight="bold",
)
f.text(
    0.055,
    0.89,
    "Data 2 | 840 pengiriman/gudang | C bermedian rendah, tetapi berekor panjang",
    fontsize=11,
)
draw_b(f, [0.04, 0.16, 0.94, 0.68], 11)
f.text(
    0.055,
    0.025,
    "Kotak: Q1–Q3 | garis: median | belah ketupat: rata-rata | whisker: 1,5 × IQR | titik: pencilan. Sumber: simulasi LKM, seed 42.",
    fontsize=10,
)
f.savefig(OUT / "kebutuhan_B.png")
plt.close(f)
f = plt.figure(figsize=(15, 5.5))
f.text(
    0.05,
    0.96,
    "E bermasalah Jumat–Sabtu sore/malam; B pada Senin pagi",
    fontsize=18,
    weight="bold",
)
f.text(
    0.05,
    0.905,
    "E: 118/120 terlambat (98,33%) pada empat sel tersebut | B: 30/30 (100%) pada Senin pagi",
    fontsize=12,
)
draw_c(f, [0.025, 0.24, 0.95, 0.66], 12)
f.text(
    0.05,
    0.025,
    "Angka sel dibulatkan ke persen utuh. Skala bersama 0–100%; warna sel menunjukkan tingkat pelanggaran, bukan identitas gudang. Sumber: simulasi LKM, seed 42.",
    fontsize=10,
)
f.savefig(OUT / "kebutuhan_C.png")
plt.close(f)
# Satu halaman laporan untuk Direktur.
f = plt.figure(figsize=(8.27, 11.69))
f.text(0.07, 0.958, "Pilih A sebagai percontohan bertahap", fontsize=20, weight="bold")
f.text(
    0.07,
    0.923,
    "A: 19,86 jam rata-rata | 14,05% melanggar SLA | 85,95% tepat waktu",
    fontsize=11.5,
)
f.text(
    0.07,
    0.90,
    "Data simulasi LKM, seed 42; 4.200 catatan. Keputusan mengutamakan SLA 24 jam.",
    fontsize=10,
)
f.text(
    0.07,
    0.866,
    "A  C unggul pada 2 dari 6 dimensi, bukan hampir semua",
    fontsize=12.5,
    weight="bold",
)
draw_a(f, [0.065, 0.652, 0.88, 0.19], 11)
f.text(
    0.07,
    0.615,
    "B  A tercepat; C memiliki risiko keterlambatan ekstrem",
    fontsize=12.5,
    weight="bold",
)
draw_b(f, [0.065, 0.438, 0.88, 0.158], 10)
f.text(
    0.07,
    0.404,
    "Kotak Q1–Q3; garis median; ◆ rata-rata; whisker 1,5 × IQR; titik pencilan.",
    fontsize=9,
)
f.text(
    0.07,
    0.381,
    "C  Risiko E terpusat Jumat–Sabtu; B pada Senin pagi",
    fontsize=12.5,
    weight="bold",
)
draw_c(f, [0.045, 0.178, 0.91, 0.185], 10)
f.text(
    0.07,
    0.117,
    "Tindakan: tambah kapasitas E pada empat shift rawan; majukan penyiapan B.",
    fontsize=10,
    weight="bold",
)
f.text(
    0.07,
    0.09,
    "Trade-off A: efisiensi biaya 70, di bawah E 88; pantau biaya per paket saat uji coba.",
    fontsize=10,
)
f.text(
    0.07,
    0.067,
    "A masih terlambat 118/840. Uji 2 minggu; target usulan ≤10% terlambat, bukan SLA baru.",
    fontsize=10,
)
f.text(
    0.07,
    0.032,
    "Sumber: Data 1 dan kode Data 2 dalam LKM. Dasar metode: NIST (box plot), OECD (agregasi),\nMatplotlib dan Seaborn. Data nyata, biaya rupiah, dan kapasitas belum tersedia.",
    fontsize=9,
)
f.savefig(OUT / "laporan_satu_halaman.png")
plt.close(f)
print("Empat visualisasi tersimpan:", OUT)

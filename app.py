# ==============================
# IMPORT LIBRARY
# ==============================

import pandas as pd
import os
import re
import uuid
import joblib
import numpy as np

from flask import (
    Flask,
    render_template,
    request
)

from datetime import datetime

# ==============================
# FLASK
# ==============================

app = Flask(__name__)

# ==============================
# BASE DIRECTORY
# ==============================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FILE_HISTORI = os.path.join(
    BASE_DIR,
    "histori_pasien.csv"
)

# ==============================
# LOAD MODEL & DATASET
# ==============================

model_rf = joblib.load(
    os.path.join(BASE_DIR, "model_rf.pkl")
)

tfidf = joblib.load(
    os.path.join(BASE_DIR, "tfidf.pkl")
)

label_encoder = joblib.load(
    os.path.join(BASE_DIR, "label_encoder.pkl")
)

dataset_model = joblib.load(
    os.path.join(BASE_DIR, "dataset.pkl")
)
print(dataset_model.columns.tolist())
# ==============================
# CLEANING TEXT
# ==============================

def clean_keluhan(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r'\d+',
        ' ',
        text
    )

    text = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()
# ==============================
# RAPIIKAN TEXT CSV
# ==============================

def rapikan_text(text):

    text = str(text)

    text = text.replace("|", ",")

    text = text.replace(" . ", ", ")
    text = text.replace(" .", "")
    text = text.replace(". ", ", ")

    text = text.replace(",,", ",")

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    text = re.sub(
        r',\s*,',
        ', ',
        text
    )

    text = text.strip()

    return text

# ==============================
# CLEAN JOIN
# ==============================

def clean_join(items):

    hasil = []

    for item in items:

        item = rapikan_text(item)

        if item != "" and item not in hasil:

            hasil.append(item)

    return ", ".join(hasil)

# ==============================
# GENERATE MASTER KELUHAN
# ==============================

def generate_keluhan_master():

    kategori_keluhan = {

        "Kepala": [],
        "Mata": [],
        "Hidung": [],
        "Tenggorokan": [],
        "Perut": [],
        "Kulit": [],
        "Kaki": [],
        "Lainnya": []
    }

    semua_keluhan = []

    if 'keluhan' in dataset_model.columns:

        keluhan_list = (

            dataset_model['keluhan']
            .dropna()
            .astype(str)
            .str.lower()
            .tolist()
        )

        semua_keluhan.extend(
            keluhan_list
        )



    semua_keluhan = sorted(
        list(
            set(semua_keluhan)
        )
    )

    mapping = {

        "Kepala": [
            "kepala",
            "pusing",
            "migrain",
            "demam"
        ],

        "Mata": [
            "mata"
        ],

        "Hidung": [
            "pilek",
            "hidung",
            "bersin"
        ],

        "Tenggorokan": [
            "batuk",
            "tenggorokan",
            "serak"
        ],

        "Perut": [
            "mual",
            "muntah",
            "diare",
            "maag",
            "perut",
            "kembung"
        ],

        "Kulit": [
            "gatal",
            "ruam",
            "kulit"
        ],

        "Kaki": [
            "kaki",
            "pegal",
            "kesemutan"
        ]
    }

    for keluhan in semua_keluhan:

        masuk = False

        for kategori, keywords in mapping.items():

            for key in keywords:

                if key in keluhan:

                    kategori_keluhan[
                        kategori
                    ].append(
                        keluhan
                    )

                    masuk = True
                    break

            if masuk:
                break

        if not masuk:

            kategori_keluhan[
                "Lainnya"
            ].append(
                keluhan
            )

    for kategori in kategori_keluhan:

        kategori_keluhan[
            kategori
        ] = sorted(

            list(

                set(
                    kategori_keluhan[
                        kategori
                    ]
                )

            )

        )

    return kategori_keluhan

# ==============================
# LOAD HISTORI
# ==============================

def load_data():

    if os.path.exists(FILE_HISTORI):

        try:

            df = pd.read_csv(
                FILE_HISTORI
            )

            kolom_text = [
            'keluhan',
            'top1',
            'top2',
            'top3',
            'obat_pendamping'
        ]

            for kolom in kolom_text:

                if kolom in df.columns:

                    df[kolom] = df[kolom].astype(str).apply(
                        rapikan_text
                    )

            return df

        except:

            return pd.DataFrame()

    return pd.DataFrame()

# ==============================
# CHART DATA
# ==============================

def get_chart_data(df):

    if not df.empty:

        obat_counts = (
            df['obat']
            .value_counts()
        )

        obat_labels = list(
            obat_counts.index
        )

        obat_values = [

            int(x)
            for x in obat_counts.values
        ]

    else:

        obat_labels = []
        obat_values = []

    return obat_labels, obat_values

# ==============================
# KETERANGAN OBAT
# ==============================

def generate_keterangan(
    keluhan,
    obat
):

    knowledge_base = {

        "paracetamol": {

            "golongan":
            "analgesik dan antipiretik",

            "fungsi":
            "membantu menurunkan demam serta meredakan nyeri ringan hingga sedang"
        },

        "amoxicillin": {

            "golongan":
            "antibiotik",

            "fungsi":
            "membantu menangani infeksi bakteri terutama pada saluran pernapasan"
        },

        "omeprazole": {

            "golongan":
            "proton pump inhibitor",

            "fungsi":
            "membantu mengurangi produksi asam lambung berlebih"
        },

        "amlodipin": {

            "golongan":
            "antihipertensi",

            "fungsi":
            "membantu mengontrol tekanan darah tinggi"
        },

        "hydrocortisone": {

            "golongan":
            "kortikosteroid",

            "fungsi":
            "membantu mengurangi peradangan dan iritasi kulit"
        }
    }

    obat_key = str(obat).lower().strip()

    if obat_key in knowledge_base:

        info = knowledge_base[obat_key]

        return (
            f"Berdasarkan hasil analisis sistem, "
            f"keluhan pasien yaitu {keluhan}. "
                 

            f"Sistem merekomendasikan penggunaan "
            f"{obat} karena termasuk golongan "
            f"{info['golongan']} yang berfungsi "
            f"{info['fungsi']}. "

            f"Rekomendasi ini diperoleh dari "
            f"proses analisis kemiripan keluhan "
            f"menggunakan metode TF-IDF dan "
            f"algoritma Random Forest pada data "
            f"historis klinik."
        )

    return (
        f"Sistem merekomendasikan {obat} "
        f"berdasarkan hasil analisis kemiripan "
        f"keluhan pasien terhadap data historis."
    )
    
# ==============================
# DASHBOARD
# ==============================

@app.route('/')
def dashboard():

    df = load_data()

    obat_labels, obat_values = (
        get_chart_data(df)
    )

    return render_template(
        "dashboard.html",

        hasil_list=None,

        total_data=len(df),

        jumlah_kelas_obat=len(
            label_encoder.classes_
        ),

        jumlah_obat=len(
            dataset_model[
                'obat_utama'
            ].unique()
        ),

        obat_labels=obat_labels,

        obat_values=obat_values,

        keluhan_master=
        generate_keluhan_master()
    )


@app.route('/pasien')

def pasien():

    df = load_data()

    data = (

        df.to_dict(
            orient='records'
        )

        if not df.empty

        else []
    )

    return render_template(

        "data_pasien.html",

        data=data
    )

# ==============================
# LAPORAN
# ==============================

@app.route('/laporan')
def laporan():

    df = load_data()

    if df.empty:

        return render_template(

            "laporan.html",

            gender_stat={},

            keluhan_stat={},

            obat_stat={},

            total=0
        )

    return render_template(

        "laporan.html",

        gender_stat=
        df['gender']
        .value_counts()
        .to_dict(),

        keluhan_stat=
        df['keluhan']
        .value_counts()
        .head(10)
        .to_dict(),

        obat_stat=
        df['obat']
        .value_counts()
        .to_dict(),

        total=len(df)
    )
# ==============================
# PRINT LAPORAN STATISTIK
# ==============================

@app.route('/print_laporan')
def print_laporan():

    df = load_data()

    if df.empty:

        return render_template(
            "laporan_print.html",
            gender_stat={},
            keluhan_stat={},
            obat_stat={},
            total=0,
            tanggal_cetak=datetime.now().strftime(
                "%d-%m-%Y"
            )
        )

    return render_template(
        "laporan_print.html",
        gender_stat=df['gender']
        .value_counts()
        .to_dict(),

        keluhan_stat=df['keluhan']
        .value_counts()
        .head(10)
        .to_dict(),

        obat_stat=df['obat']
        .value_counts()
        .to_dict(),

        total=len(df),

        tanggal_cetak=datetime.now().strftime(
            "%d-%m-%Y"
        )
    )

# ==============================
# PRINT LAPORAN PASIEN
# ==============================

@app.route('/print/<int:index>')
def print_data(index):

    df = load_data()

    if df.empty:
        return "Data pasien kosong"

    if index < 0 or index >= len(df):
        return "Data pasien tidak ditemukan"

    p = df.iloc[index]

    # ==========================
    # AMBIL DATA HISTORI
    # ==========================

    keluhan_text = str(p.get('keluhan', '-'))
    obat_text = str(p.get('obat', '-'))
    pendamping_text = str(
        p.get('obat_pendamping', '-')
    )

    daftar_keluhan = [
        x.strip()
        for x in keluhan_text.split(',')
        if x.strip()
    ]


    daftar_obat = [
        x.strip()
        for x in obat_text.split(',')
        if x.strip()
    ]

    # Obat pendamping ditampilkan apa adanya
    obat_pendamping = pendamping_text

    jumlah = max(
        len(daftar_keluhan),
        len(daftar_obat)
    )

    hasil_list = []

    for i in range(jumlah):

        keluhan = (
            daftar_keluhan[i]
            if i < len(daftar_keluhan)
            else "-"
        )


        obat = (
            daftar_obat[i]
            if i < len(daftar_obat)
            else "-"
        )
        hasil_list.append({

            "keluhan": keluhan,

            "diagnosa": p.get(
                'diagnosa',
                '-'
            ),

            "obat": obat,

            "obat_pendamping":
            obat_pendamping,

            "prob":100,

            "keterangan":
            generate_keterangan(
                keluhan,
                obat
            )
        })

    return render_template(

        "laporan_pasien.html",

        p=p,

        hasil_list=hasil_list

    )

   
@app.route('/predict', methods=['POST'])
def predict():

    from collections import Counter
    from scipy.sparse import hstack, csr_matrix

    nama = request.form['nama']
    gender = request.form['gender']
    usia = int(request.form['usia'])

    keluhan_list = request.form.getlist('keluhan')

    if len(keluhan_list) == 0:

        return render_template(
            "dashboard.html",
            hasil_list=[],
            error="Pilih minimal 1 keluhan",
            total_data=len(load_data()),
            jumlah_kelas_obat=len(
                label_encoder.classes_
            ),
            jumlah_obat=len(
                dataset_model['obat_utama'].unique()
            ),
            keluhan_master=
            generate_keluhan_master()
        )

    # ==================================
    # PREPROCESS INPUT
    # ==================================

    keluhan_text = ", ".join(
        keluhan_list
    )

    keluhan_bersih = clean_keluhan(
        keluhan_text
    )

    gender_encode = (
        1 if gender == "laki-laki"
        else 0
    )

    X_text = tfidf.transform(
        [keluhan_bersih]
    )

    fitur_tambahan = csr_matrix(
        [[usia, gender_encode]]
    )

    X_input = hstack([
        X_text,
        fitur_tambahan
    ])

    # ==================================
    # PREDIKSI RANDOM FOREST
    # ==================================

    probabilitas = model_rf.predict_proba(
        X_input
    )[0]

    top_idx = np.argsort(
        probabilitas
    )[-3:][::-1]

    top_obat = label_encoder.inverse_transform(
        top_idx
    )

    # ==================================
    # HASIL PER KELUHAN
    # ==================================

    hasil_list = []

    for i, keluhan_input in enumerate(
        keluhan_list
    ):

        # ==========================
        # TOP OBAT
        # ==========================

        idx_obat = top_idx[
            i if i < len(top_idx)
            else 0
        ]

        obat = label_encoder.inverse_transform(
            [idx_obat]
        )[0]

        prob = round(
            probabilitas[idx_obat] * 100,
            2
        )

        # ==========================
        # HISTORI KELUHAN
        # ==========================

        keluhan_clean = clean_keluhan(
            keluhan_input
        )

        kata_kunci = "|".join(
            keluhan_clean.split()
        )

        histori_keluhan = dataset_model[
            dataset_model['keluhan']
            .astype(str)
            .str.lower()
            .str.contains(
                kata_kunci,
                regex=True,
                na=False
            )
        ]

        keluhan_mirip = keluhan_input

        if len(histori_keluhan) > 0:

            keluhan_mirip = str(
                histori_keluhan.iloc[0][
                    'keluhan'
                ]
            )

        # ==========================
        # OBAT PENDAMPING
        # ==========================

        daftar_pendamping = []

        if len(histori_keluhan) > 0:

            for resep in histori_keluhan[
                'pengobatan/resep'
            ]:

                resep_split = [

                    x.strip()

                    for x in re.split(
                        r'[|,;/]',
                        str(resep)
                    )

                    if x.strip()
                ]

                if len(resep_split) > 1:

                    daftar_pendamping.extend(
                        resep_split[1:]
                    )

        counter = Counter(
            daftar_pendamping
        )

        top_pendamping = [

            obat

            for obat, _
            in counter.most_common(3)
        ]

        if len(top_pendamping) > 0:

            obat_pendamping_item = (
                ", ".join(
                    top_pendamping
                )
            )

        else:

            obat_pendamping_item = "-"

        print("\n====================")
        print("KELUHAN :", keluhan_input)
        print("MIRIP :", keluhan_mirip)
        print("OBAT :", obat)
        print("PENDAMPING :", obat_pendamping_item)
        print("====================")

        hasil_list.append({

            "keluhan":
            keluhan_input,

            "keluhan_mirip":
            keluhan_mirip,

            "obat":
            obat,

            "prob":
            prob,

            "obat_pendamping":
            obat_pendamping_item,

            "keterangan":
            generate_keterangan(
                keluhan_input,
                obat
            )
        })

    # ==================================
    # GABUNG PENDAMPING UNTUK HISTORI
    # ==================================

    obat_pendamping_histori = clean_join(

        [

            item["obat_pendamping"]

            for item in hasil_list

            if item[
                "obat_pendamping"
            ] != "-"

        ]
    )

    if obat_pendamping_histori == "":

        obat_pendamping_histori = "-"

    # ==================================
    # SIMPAN HISTORI
    # ==================================

    histori_baru = pd.DataFrame([{

        "id":
        str(uuid.uuid4())[:8],

        "tanggal":
        datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        ),

        "nama":
        nama,

        "gender":
        gender,

        "usia":
        usia,

        "keluhan":
        keluhan_text,

        "top1":
        top_obat[0],

        "top2":
        top_obat[1],

        "top3":
        top_obat[2],

        "obat":
        top_obat[0],

        "obat_pendamping":
        obat_pendamping_histori

    }])

    if os.path.exists(
        FILE_HISTORI
    ):

        histori_lama = pd.read_csv(
            FILE_HISTORI
        )

        histori_baru = pd.concat(
            [
                histori_lama,
                histori_baru
            ],
            ignore_index=True
        )

    histori_baru.to_csv(
        FILE_HISTORI,
        index=False
    )

    # ==================================
    # RETURN
    # ==================================

    return render_template(

        "dashboard.html",

        hasil_list=
        hasil_list,

        nama=
        nama,

        gender=
        gender,

        usia=
        usia,

        keluhan=
        keluhan_text,

        total_data=
        len(load_data()),

        jumlah_kelas_obat=
        len(
            label_encoder.classes_
        ),

        jumlah_obat=
        len(
            dataset_model[
                'obat_utama'
            ].unique()
        ),

        keluhan_master=
        generate_keluhan_master()
    )
# ==============================
# MAIN
# ==============================

if __name__ == '__main__':

    app.run(
        debug=True
    )
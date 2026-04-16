from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


# =============================================================================
# CONFIG + KONSTANTA
# =============================================================================

st.set_page_config(
    page_title="Dashboard Gizi NTB",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main { padding-top: 1.2rem; }
    h1, h2, h3 { letter-spacing: 0.2px; }
    .debug-note {
        background: #f7f7f7;
        border-left: 4px solid #666;
        padding: 0.5rem 0.75rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).parent
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
INDICATORS = ["stunting", "underweight", "wasting"]
IMPUTED_LABEL = "Imputed (MICE 2020)"

# 10 wilayah resmi NTB (kabupaten/kota)
OFFICIAL_NTB_REGIONS = [
    "Bima",
    "Dompu",
    "Kota Bima",
    "Lombok Barat",
    "Lombok Tengah",
    "Lombok Timur",
    "Lombok Utara",
    "Mataram",
    "Sumbawa",
    "Sumbawa Barat",
]

INDICATOR_LABELS = {
    "stunting": "Stunting",
    "underweight": "Underweight",
    "wasting": "Wasting",
}

INDICATOR_DESCRIPTIONS = {
    "stunting": "Tinggi badan menurut umur (HAZ < -2 SD)",
    "underweight": "Berat badan menurut umur (WAZ < -2 SD)",
    "wasting": "Berat badan menurut tinggi badan (WHZ < -2 SD)",
}

COLOR_SCHEME = {
    "stunting": "#EF553B",      # merah
    "underweight": "#FF9900",   # oranye
    "wasting": "#00CC96",       # hijau
}

# Untuk peta: rendah=Hijau, sedang=Kuning, tinggi=Merah
MAP_RYG_SCALE = [
    (0.0, "#2E7D32"),   # hijau
    (0.5, "#FDD835"),   # kuning
    (1.0, "#C62828"),   # merah
]

COLOR_SCALE = {
    "stunting": px.colors.sequential.Reds,
    "underweight": px.colors.sequential.Oranges,
    "wasting": px.colors.sequential.Greens,
}

DATA_SOURCES = [
    {
        "dataset": "Persentase Stunting, Underweight, Wasting Kabupaten/Kota",
        "instansi": "Kementerian Sekretariat Negara RI (Dashboard Stunting)",
        "cakupan": "2018-2024",
        "url": "https://dashboard.stunting.go.id/masalah-gizi-pada-balita-kabupaten/",
    },
    {
        "dataset": "Persentase Stunting, Underweight, Wasting Provinsi",
        "instansi": "Kementerian Sekretariat Negara RI (Dashboard Stunting)",
        "cakupan": "2018-2024",
        "url": "https://dashboard.stunting.go.id/masalah-gizi-pada-balita/",
    },
    {
        "dataset": "Persentase Ruta yang memiliki akses terhadap sanitasi layak",
        "instansi": "BPS Provinsi Nusa Tenggara Barat",
        "cakupan": "2015-2024 (disesuaikan pemakaian dashboard)",
        "url": "https://ntb.bps.go.id/en/statistics-table/3/VGtGTU5qbDFlQzl1VWxCTVNWZElXbWRhWkUwMFVUMDkjMw==/persentase-rumah-tangga-yang-memiliki-akses-terhadap-sanitasi-layak-menurut-kabupaten-kota-di-provinsi-nusa-tenggara-barat--2015.html",
    },
    {
        "dataset": "Jumlah Penduduk Miskin menurut Kabupaten/Kota",
        "instansi": "BPS Provinsi Nusa Tenggara Barat",
        "cakupan": "Mengikuti tabel resmi BPS NTB",
        "url": "https://ntb.bps.go.id/id/statistics-table/2/MjI1IzI=/jumlah-penduduk-miskin-menurut-kabupaten-kota.html",
    },
    {
        "dataset": "Jumlah Penduduk menurut Kabupaten/Kota",
        "instansi": "BPS Provinsi Nusa Tenggara Barat",
        "cakupan": "Termasuk publikasi 2024",
        "url": "https://ntb.bps.go.id/id/statistics-table/3/V1ZSbFRUY3lTbFpEYTNsVWNGcDZjek53YkhsNFFUMDkjMw==/penduduk--laju-pertumbuhan-penduduk--distribusi-persentase-penduduk--kepadatan-penduduk--rasio-jenis-kelamin-penduduk-menurut-kabupaten-kota-di-provinsi-nusa-tenggara-barat--2024.html?year=2024",
    },
    {
        "dataset": "Persentase Pelayanan Kesehatan Bayi",
        "instansi": "Satu Data NTB",
        "cakupan": "Mengikuti dataset resmi",
        "url": "https://data.ntbprov.go.id/dataset/9d27c8bd-6883-4ae3-a38e-7b486c9f46ba/show",
    },
    {
        "dataset": "Indeks Pembangunan Manusia (IPM)",
        "instansi": "Satu Data NTB",
        "cakupan": "Mengikuti dataset resmi",
        "url": "https://data.ntbprov.go.id/dataset/9ef87836-5138-4839-a7fa-b611af5f5d44/show",
    },
]


# =============================================================================
# HELPERS - NORMALISASI, LOADING, PREPROCESSING
# =============================================================================


def base_layout() -> Dict:
    # Template dasar untuk semua chart Plotly agar tampilannya konsisten.
    return {
        # Tema putih untuk latar yang bersih dan mudah dibaca.
        "template": "plotly_white",
        # Font umum agar semua chart punya tipografi yang sama.
        "font": {"family": "Arial, sans-serif", "size": 12},
        # Hover mode unified supaya tooltip lebih ringkas dan seragam.
        "hovermode": "x unified",
        # Margin chart agar elemen tidak terlalu mepet ke tepi.
        "margin": {"l": 50, "r": 20, "t": 55, "b": 50},
        # Tinggi default chart agar proporsinya stabil.
        "height": 420,
    }


def normalize_region_name(name: str) -> str:
    # Jika input kosong, kembalikan string kosong.
    if pd.isna(name):
        return ""

    # Ubah input menjadi string bersih dan hilangkan spasi berlebih.
    text = str(name).strip()

    # Hapus prefix umum yang sering muncul di nama wilayah.
    text = text.replace("Kab.", "").replace("Kabupaten", "").strip()

    # Perbaiki kasus khusus nama kota agar konsisten.
    lower = text.lower()
    if lower in {"kota bima", "bima kota"}:
        return "Kota Bima"
    if lower in {"kota mataram", "mataram"}:
        return "Mataram"
    if lower.startswith("kota "):
        text = text[5:].strip()

    # Rapikan spasi lalu ubah ke title case.
    text = " ".join(text.split())
    text = text.title()

    # Keluarkan nama wilayah yang sudah distandardisasi.
    return text


def parse_number_localized(value) -> float:
    """Parse angka format Indonesia/locale campuran ke float.

    Contoh yang didukung:
    - 1.192.110,0 -> 1192110.0
    - 1 067,7 -> 1067.7
    - 105.24 -> 105.24
    """
    # Nilai kosong tetap dianggap NaN.
    if pd.isna(value):
        return np.nan

    # Kalau sudah numerik, langsung ubah ke float.
    if isinstance(value, (int, float, np.number)):
        return float(value)

    # Bersihkan string dari spasi dan karakter aneh.
    s = str(value).strip()
    if s == "":
        return np.nan

    # Format Indonesia memakai titik ribuan dan koma desimal.
    s = s.replace(" ", "")
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")

    # Sisakan karakter angka, minus, dan titik saja.
    s = re.sub(r"[^0-9\.-]", "", s)
    if s in {"", ".", "-", "-."}:
        return np.nan

    # Konversi akhir ke float, kalau gagal kembalikan NaN.
    try:
        return float(s)
    except ValueError:
        return np.nan


def standardize_wide_metric(df_raw: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Ubah tabel wide (tahun sebagai kolom) menjadi long: kabupaten, tahun, value_col."""
    # Salin data agar data asli tidak berubah.
    df = df_raw.copy()
    # Seragamkan nama kolom supaya mudah dicari.
    df.columns = [str(c).strip() for c in df.columns]

    # Coba temukan kolom wilayah dari beberapa kandidat nama umum.
    region_candidates = {"kabupaten/kota", "kabupaten", "kota/kabupaten", "wilayah"}
    region_col = None
    for c in df.columns:
        if str(c).strip().lower() in region_candidates:
            region_col = c
            break

    # Kalau tidak ketemu, gunakan kolom pertama sebagai fallback.
    if region_col is None:
        region_col = df.columns[0]

    # Ubah bentuk data dari wide ke long.
    year_cols = [c for c in df.columns if c != region_col]
    out = df.melt(id_vars=[region_col], value_vars=year_cols, var_name="tahun", value_name=value_col)
    out = out.rename(columns={region_col: "kabupaten"})

    # Standardisasi nama wilayah, tahun, dan isi nilainya.
    out["kabupaten"] = out["kabupaten"].map(normalize_region_name)
    out["tahun"] = pd.to_numeric(out["tahun"], errors="coerce").astype("Int64")
    out[value_col] = out[value_col].map(parse_number_localized)

    # Buang baris yang tidak punya wilayah atau tahun valid.
    return out.dropna(subset=["kabupaten", "tahun"])


def _read_table_with_optional_xlsx(stem_name: str) -> pd.DataFrame:
    """Baca hanya file .xlsx sesuai permintaan."""
    # Bentuk path file Excel berdasarkan nama dasar.
    xlsx_path = BASE_DIR / f"{stem_name}.xlsx"

    # Jika file ada, baca langsung.
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)

    # Kalau tidak ada, berikan error yang jelas.
    raise FileNotFoundError(f"File tidak ditemukan: {xlsx_path.name}")


def standardize_kabupaten_indicator(df_raw: pd.DataFrame, indicator: str) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "kabupaten/kota": "kabupaten",
        "kabupaten": "kabupaten",
        "tahun": "tahun",
        "persentase": indicator,
    }
    df = df.rename(columns=rename_map)

    required = {"kabupaten", "tahun", indicator}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Kolom data {indicator} belum lengkap, kurang: {sorted(missing)}")

    df = df[["kabupaten", "tahun", indicator]].copy()
    df["kabupaten"] = df["kabupaten"].map(normalize_region_name)
    df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce").astype("Int64")
    df[indicator] = pd.to_numeric(df[indicator], errors="coerce")

    return df.dropna(subset=["kabupaten", "tahun"])


def standardize_provinsi_indicator(df_raw: pd.DataFrame, indicator: str) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "tahun": "tahun",
        "persentase": indicator,
    }
    df = df.rename(columns=rename_map)

    required = {"tahun", indicator}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Kolom data provinsi {indicator} belum lengkap, kurang: {sorted(missing)}")

    df = df[["tahun", indicator]].copy()
    df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce").astype("Int64")
    df[indicator] = pd.to_numeric(df[indicator], errors="coerce")

    return df.dropna(subset=["tahun"])


def linear_impute(series: pd.Series) -> pd.Series:
    """Imputasi linear Y = a + b*tahun, fallback mean jika data valid < 2."""
    y = series.copy()
    if y.isna().sum() == 0:
        return y

    valid = y.dropna()
    if len(valid) < 2:
        fill_value = valid.mean() if len(valid) > 0 else 0.0
        return y.fillna(fill_value)

    x_valid = valid.index.to_numpy(dtype=float)
    y_valid = valid.to_numpy(dtype=float)
    b, a = np.polyfit(x_valid, y_valid, 1)  # y = b*x + a

    x_all = y.index.to_numpy(dtype=float)
    pred = b * x_all + a
    y[y.isna()] = pred[y.isna()]
    return y


def _build_imputed_segment_mask(years: pd.Series, imputed_mask: pd.Series) -> pd.Series:
    """Ambil segmen sekitar titik imputasi (tahun t-1, t, t+1) untuk garis putus-putus."""
    if imputed_mask.sum() == 0:
        return pd.Series(False, index=years.index)

    imp_years = set(years.loc[imputed_mask].astype(int).tolist())
    seg_years = set(imp_years)
    for y in imp_years:
        seg_years.add(y - 1)
        seg_years.add(y + 1)
    return years.astype(int).isin(sorted(seg_years))


def impute_indicator_2020_with_mice(
    df: pd.DataFrame,
    indicator: str,
    predictor_cols: List[str],
    group_col: str = "kabupaten",
) -> Tuple[pd.Series, pd.Series]:
    """Imputasi nilai indikator tahun 2020 dengan MICE (RandomForestRegressor)."""
    y_filled = df[indicator].copy()
    imputed_flag = pd.Series(False, index=df.index)

    target_mask = (df["tahun"].astype(int) == 2020) & y_filled.isna()
    if not target_mask.any():
        return y_filled, imputed_flag

    matrix = df[[indicator] + predictor_cols].copy()
    matrix["tahun_num"] = pd.to_numeric(df["tahun"], errors="coerce")
    matrix["group_code"] = df[group_col].astype("category").cat.codes.astype(float)

    for c in matrix.columns:
        matrix[c] = pd.to_numeric(matrix[c], errors="coerce")

    try:
        estimator = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=1,
        )
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=25,
            random_state=42,
            initial_strategy="median",
        )
        imputed_matrix = imputer.fit_transform(matrix)
        pred = pd.Series(imputed_matrix[:, 0], index=df.index).clip(0.0, 100.0)

        y_filled.loc[target_mask] = pred.loc[target_mask]
        imputed_flag.loc[target_mask] = True
    except Exception:
        # Fallback jika MICE gagal: linear interpolation per wilayah.
        pass

    unresolved_mask = (df["tahun"].astype(int) == 2020) & y_filled.isna()
    if unresolved_mask.any():
        for region in df.loc[unresolved_mask, group_col].dropna().unique().tolist():
            mask_region = df[group_col] == region
            s = pd.Series(
                y_filled.loc[mask_region].values,
                index=df.loc[mask_region, "tahun"].astype(int).values,
            )
            filled = linear_impute(s).clip(0.0, 100.0)

            idx = df.index[mask_region]
            before = y_filled.loc[idx].copy()
            y_filled.loc[idx] = filled.values
            imputed_flag.loc[idx] = (
                (df.loc[idx, "tahun"].astype(int) == 2020)
                & before.isna()
                & y_filled.loc[idx].notna()
            )

    return y_filled, imputed_flag


def impute_prov_2020_with_mice(
    df: pd.DataFrame,
    indicator: str,
    predictor_cols: List[str],
) -> Tuple[pd.Series, pd.Series]:
    """Imputasi indikator level provinsi tahun 2020 dengan MICE."""
    y_filled = df[indicator].copy()
    imputed_flag = pd.Series(False, index=df.index)

    target_mask = (df["tahun"].astype(int) == 2020) & y_filled.isna()
    if not target_mask.any():
        return y_filled, imputed_flag

    matrix = df[[indicator] + predictor_cols].copy()
    matrix["tahun_num"] = pd.to_numeric(df["tahun"], errors="coerce")
    for c in matrix.columns:
        matrix[c] = pd.to_numeric(matrix[c], errors="coerce")

    try:
        estimator = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=1,
        )
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=25,
            random_state=42,
            initial_strategy="median",
        )
        imputed_matrix = imputer.fit_transform(matrix)
        pred = pd.Series(imputed_matrix[:, 0], index=df.index).clip(0.0, 100.0)

        y_filled.loc[target_mask] = pred.loc[target_mask]
        imputed_flag.loc[target_mask] = True
    except Exception:
        # Fallback jika MICE gagal: linear interpolation time-series provinsi.
        s = pd.Series(y_filled.values, index=df["tahun"].astype(int).values)
        before = y_filled.copy()
        y_filled = linear_impute(s).clip(0.0, 100.0)
        y_filled = pd.Series(y_filled.values, index=df.index)
        imputed_flag = ((df["tahun"].astype(int) == 2020) & before.isna() & y_filled.notna())

    return y_filled, imputed_flag


def preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict, List[str]]:
    """
    Returns:
        df_kabupaten_merged, df_provinsi_merged, geojson_std, available_regions
    """
    # --- load 6 file data ---
    kab_st = standardize_kabupaten_indicator(_read_table_with_optional_xlsx("data_stunting_kabupaten"), "stunting")
    kab_uw = standardize_kabupaten_indicator(_read_table_with_optional_xlsx("data_underweight_kabupaten"), "underweight")
    kab_ws = standardize_kabupaten_indicator(_read_table_with_optional_xlsx("data_wasting_kabupaten"), "wasting")

    prov_st = standardize_provinsi_indicator(_read_table_with_optional_xlsx("data_stunting_provinsi"), "stunting")
    prov_uw = standardize_provinsi_indicator(_read_table_with_optional_xlsx("data_underweight_provinsi"), "underweight")
    prov_ws = standardize_provinsi_indicator(_read_table_with_optional_xlsx("data_wasting_provinsi"), "wasting")

    # --- merge kabupaten ---
    df_kab = (
        kab_st.merge(kab_uw, on=["kabupaten", "tahun"], how="outer")
        .merge(kab_ws, on=["kabupaten", "tahun"], how="outer")
    )

    # filter tahun target
    df_kab = df_kab[df_kab["tahun"].isin(YEARS)].copy()

    # --- load & normalisasi geojson ---
    geojson_path = BASE_DIR / "Kabupaten-Kota (Provinsi Nusa Tenggara Barat).geojson"
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON tidak ditemukan: {geojson_path}")

    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    # key properti nama wilayah (fallback fleksibel)
    feature_keys_priority = ["nama", "NAME_2", "name", "NAME"]

    def get_feature_name(props: Dict) -> str:
        for k in feature_keys_priority:
            if k in props and str(props.get(k, "")).strip() != "":
                return str(props[k])
        return ""

    geo_regions = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        region = normalize_region_name(get_feature_name(props))

        # kasus khusus agar selaras daftar resmi
        if region == "Bima" and str(props.get("TYPE_2", "")).lower() == "city":
            region = "Kota Bima"

        props["kabupaten_std"] = region
        geo_regions.append(region)

    geo_regions = sorted({r for r in geo_regions if r})

    # pakai intersection antara daftar resmi + geojson (harus tetap 10 by design)
    region_whitelist = sorted(set(OFFICIAL_NTB_REGIONS).intersection(set(geo_regions)))
    if len(region_whitelist) == 0:
        region_whitelist = OFFICIAL_NTB_REGIONS.copy()

    # filter hanya 10 wilayah resmi
    df_kab = df_kab[df_kab["kabupaten"].isin(region_whitelist)].copy()

    # lakukan full grid (kabupaten x tahun) agar imputasi konsisten
    full_idx = pd.MultiIndex.from_product(
        [region_whitelist, YEARS], names=["kabupaten", "tahun"]
    )
    df_kab = (
        df_kab.set_index(["kabupaten", "tahun"]) 
        .reindex(full_idx)
        .reset_index()
        .sort_values(["kabupaten", "tahun"])
    )

    # --- data baru: populasi, penduduk miskin, sanitasi, ipm, pelayanan kesehatan ---
    pop_df = standardize_wide_metric(_read_table_with_optional_xlsx("jumlah_penduduk"), "jumlah_penduduk")
    poor_df = standardize_wide_metric(_read_table_with_optional_xlsx("jumlah_penduduk_miskin"), "jumlah_penduduk_miskin")
    sanit_df = standardize_wide_metric(_read_table_with_optional_xlsx("persentase_akses_sanitasi"), "akses_sanitasi")
    ipm_df = standardize_wide_metric(_read_table_with_optional_xlsx("ipm_provinsi_kab"), "ipm")
    pelkes_df = standardize_wide_metric(
        _read_table_with_optional_xlsx("persentase_pelayanan_kesehatan_bayi"),
        "pelayanan_kesehatan",
    )

    # normalisasi satuan populasi (beberapa tahun tercatat dalam skala x1000)
    pop_df["jumlah_penduduk"] = np.where(
        pop_df["jumlah_penduduk"] > 10000,
        pop_df["jumlah_penduduk"] / 1000.0,
        pop_df["jumlah_penduduk"],
    )

    # filter domain wilayah/tahun dashboard
    pop_df = pop_df[(pop_df["kabupaten"].isin(region_whitelist)) & (pop_df["tahun"].isin(YEARS))]
    poor_df = poor_df[(poor_df["kabupaten"].isin(region_whitelist)) & (poor_df["tahun"].isin(YEARS))]
    sanit_df = sanit_df[(sanit_df["kabupaten"].isin(region_whitelist)) & (sanit_df["tahun"].isin(YEARS))]
    ipm_df = ipm_df[ipm_df["tahun"].isin(YEARS)].copy()
    pelkes_df = pelkes_df[pelkes_df["tahun"].isin(YEARS)].copy()

    # hitung persentase kemiskinan = (jumlah penduduk miskin / jumlah penduduk) * 100
    poverty_df = poor_df.merge(pop_df, on=["kabupaten", "tahun"], how="inner")
    poverty_df = poverty_df[poverty_df["jumlah_penduduk"] > 0].copy()
    poverty_df["persen_kemiskinan"] = (
        poverty_df["jumlah_penduduk_miskin"] / poverty_df["jumlah_penduduk"]
    ) * 100.0
    poverty_df = poverty_df[["kabupaten", "tahun", "persen_kemiskinan"]]

    # imputasi sanitasi 2024 dengan regresi linear per kabupaten
    sanit_full_idx = pd.MultiIndex.from_product(
        [region_whitelist, YEARS], names=["kabupaten", "tahun"]
    )
    sanit_df = (
        sanit_df.set_index(["kabupaten", "tahun"])
        .reindex(sanit_full_idx)
        .reset_index()
        .sort_values(["kabupaten", "tahun"])
    )
    for region in region_whitelist:
        mask = sanit_df["kabupaten"] == region
        s = pd.Series(
            sanit_df.loc[mask, "akses_sanitasi"].values,
            index=sanit_df.loc[mask, "tahun"].values,
        )
        sanit_df.loc[mask, "akses_sanitasi"] = linear_impute(s).values

    # gabungkan metrik baru ke data utama kabupaten
    df_kab = df_kab.merge(poverty_df, on=["kabupaten", "tahun"], how="left")
    df_kab = df_kab.merge(
        sanit_df[["kabupaten", "tahun", "akses_sanitasi"]],
        on=["kabupaten", "tahun"],
        how="left",
    )
    df_kab = df_kab.merge(
        ipm_df[ipm_df["kabupaten"].isin(region_whitelist)][["kabupaten", "tahun", "ipm"]],
        on=["kabupaten", "tahun"],
        how="left",
    )
    df_kab = df_kab.merge(
        pelkes_df[pelkes_df["kabupaten"].isin(region_whitelist)][["kabupaten", "tahun", "pelayanan_kesehatan"]],
        on=["kabupaten", "tahun"],
        how="left",
    )

    # imputasi indikator tahun 2020 menggunakan MICE (fitur: sanitasi, kemiskinan, IPM, pelayanan + encoding).
    mice_predictors = ["akses_sanitasi", "persen_kemiskinan", "ipm", "pelayanan_kesehatan"]
    for ind in INDICATORS:
        df_kab[ind], df_kab[f"{ind}_imputed"] = impute_indicator_2020_with_mice(
            df_kab,
            indicator=ind,
            predictor_cols=mice_predictors,
            group_col="kabupaten",
        )

    # Jaga-jaga: isi nilai kosong selain target MICE dengan interpolation per kabupaten.
    for region in region_whitelist:
        mask = df_kab["kabupaten"] == region
        for ind in INDICATORS:
            if df_kab.loc[mask, ind].isna().any():
                s = pd.Series(df_kab.loc[mask, ind].values, index=df_kab.loc[mask, "tahun"].values)
                df_kab.loc[mask, ind] = linear_impute(s).clip(0.0, 100.0).values

    # --- merge provinsi ---
    df_prov = (
        prov_st.merge(prov_uw, on="tahun", how="outer")
        .merge(prov_ws, on="tahun", how="outer")
    )
    df_prov = df_prov[df_prov["tahun"].isin(YEARS)].sort_values("tahun")
    df_prov = pd.DataFrame({"tahun": YEARS}).merge(df_prov, on="tahun", how="left")

    # fallback jika provinsi kosong/kurang: hitung dari rata-rata kabupaten
    prov_from_kab = df_kab.groupby("tahun", as_index=False)[INDICATORS].mean()
    df_prov = prov_from_kab.merge(df_prov, on="tahun", how="left", suffixes=("_kab", "_src"))
    for ind in INDICATORS:
        df_prov[ind] = df_prov[f"{ind}_src"]

    # predictor level provinsi untuk MICE (prioritas data provinsi jika tersedia).
    prov_name = "Nusa Tenggara Barat"
    ipm_prov = ipm_df[ipm_df["kabupaten"] == prov_name][["tahun", "ipm"]].copy()
    pelkes_prov = pelkes_df[pelkes_df["kabupaten"] == prov_name][["tahun", "pelayanan_kesehatan"]].copy()
    prov_aux = df_kab.groupby("tahun", as_index=False)[["persen_kemiskinan", "akses_sanitasi", "ipm", "pelayanan_kesehatan"]].mean()
    prov_aux = prov_aux.merge(ipm_prov, on="tahun", how="left", suffixes=("_kab", "_prov"))
    prov_aux["ipm"] = prov_aux["ipm_prov"].combine_first(prov_aux["ipm_kab"])
    prov_aux = prov_aux.drop(columns=["ipm_kab", "ipm_prov"])
    prov_aux = prov_aux.merge(pelkes_prov, on="tahun", how="left", suffixes=("_kab", "_prov"))
    prov_aux["pelayanan_kesehatan"] = prov_aux["pelayanan_kesehatan_prov"].combine_first(prov_aux["pelayanan_kesehatan_kab"])
    prov_aux = prov_aux.drop(columns=["pelayanan_kesehatan_kab", "pelayanan_kesehatan_prov"])

    df_prov = df_prov.merge(prov_aux, on="tahun", how="left")
    for ind in INDICATORS:
        df_prov[ind], df_prov[f"{ind}_imputed"] = impute_prov_2020_with_mice(
            df_prov,
            indicator=ind,
            predictor_cols=["akses_sanitasi", "persen_kemiskinan", "ipm", "pelayanan_kesehatan"],
        )
        df_prov[ind] = df_prov[ind].combine_first(df_prov[f"{ind}_kab"])  # fallback final dari rerata kabupaten

    df_prov = df_prov[["tahun"] + INDICATORS + [f"{i}_imputed" for i in INDICATORS]].sort_values("tahun")

    # kategori bonus
    for ind in INDICATORS:
        q1, q2 = df_kab[ind].quantile([0.33, 0.66]).values
        df_kab[f"{ind}_kategori"] = pd.cut(
            df_kab[ind],
            bins=[-np.inf, q1, q2, np.inf],
            labels=["Rendah", "Sedang", "Tinggi"],
        )

    return df_kab, df_prov, geojson, region_whitelist


@st.cache_data
def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, Dict, List[str]]:
    return preprocess_data()


# =============================================================================
# HELPERS - CHARTS
# =============================================================================


def create_choropleth(df_kab: pd.DataFrame, geojson: Dict, indicator: str, period_label: str, multi_year: bool = False) -> go.Figure:
    d = df_kab.copy()
    d = d.rename(columns={"kabupaten": "kabupaten_std"})

    fig = px.choropleth(
        d,
        geojson=geojson,
        locations="kabupaten_std",
        featureidkey="properties.kabupaten_std",
        color=indicator,
        color_continuous_scale=MAP_RYG_SCALE,
        hover_name="kabupaten_std",
        hover_data={"kabupaten_std": False},
        labels={indicator: f"{INDICATOR_LABELS[indicator]} (%)"},
        title=(
            f"Peta Rata-rata {INDICATOR_LABELS[indicator]} - Periode {period_label}"
            if multi_year
            else f"Peta {INDICATOR_LABELS[indicator]} - Tahun {period_label}"
        ),
    )

    fig.update_traces(
        customdata=d[["kabupaten_std"]].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            + f"{INDICATOR_LABELS[indicator]}: %{{z:.2f}}%"
            + "<extra></extra>"
        ),
    )

    layout = base_layout()
    layout.update(
        {
            "geo": {
                "projection_type": "mercator",
                "showcoastlines": True,
                "fitbounds": "locations",
            },
            "coloraxis_colorbar": {
                "title": f"{INDICATOR_LABELS[indicator]} (%)",
                "thickness": 18,
            },
        }
    )
    fig.update_layout(**layout)
    return fig


def create_line_provinsi(df_prov: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for ind in INDICATORS:
        fig.add_trace(
            go.Scatter(
                x=df_prov["tahun"],
                y=df_prov[ind],
                mode="lines+markers",
                name=INDICATOR_LABELS[ind],
                line={"color": COLOR_SCHEME[ind], "width": 2.2},
                marker={"size": 8},
            )
        )

        imp_col = f"{ind}_imputed"
        if imp_col in df_prov.columns and df_prov[imp_col].fillna(False).any():
            seg_mask = _build_imputed_segment_mask(df_prov["tahun"], df_prov[imp_col].fillna(False))
            seg = df_prov.copy()
            seg.loc[~seg_mask, ind] = np.nan
            seg_symbol = ["diamond" if bool(v) else "circle-open" for v in seg[imp_col].fillna(False)]
            fig.add_trace(
                go.Scatter(
                    x=seg["tahun"],
                    y=seg[ind],
                    mode="lines+markers",
                    name=f"{INDICATOR_LABELS[ind]} — {IMPUTED_LABEL}",
                    line={"color": COLOR_SCHEME[ind], "width": 2.8, "dash": "dash"},
                    marker={"size": 9, "symbol": seg_symbol},
                    legendgroup=f"{ind}_imputed",
                )
            )

    layout = base_layout()
    layout.update({"title": "Tren Ketiga Indikator (Provinsi)", "xaxis_title": "Tahun", "yaxis_title": "Persentase (%)"})
    fig.update_layout(**layout)
    return fig


def create_bar_kabupaten(df_kab: pd.DataFrame, indikator: str, period_label: str, top_n: int) -> go.Figure:
    d = df_kab.sort_values(indikator, ascending=False).head(top_n)
    d = d.sort_values(indikator, ascending=True)

    title_suffix = f"Periode {period_label}" if is_aggregated_period(df_kab) else f"Tahun {period_label}"

    fig = px.bar(
        d,
        x=indikator,
        y="kabupaten",
        orientation="h",
        color=indikator,
        color_continuous_scale=COLOR_SCALE[indikator],
        text=indikator,
        labels={indikator: f"{INDICATOR_LABELS[indikator]} (%)", "kabupaten": "Kabupaten/Kota"},
        title=f"Ranking {INDICATOR_LABELS[indikator]} - {title_suffix}",
    )
    fig.update_traces(texttemplate="%{x:.2f}%", textposition="outside")

    layout = base_layout()
    layout.update({"showlegend": False, "xaxis_title": f"{INDICATOR_LABELS[indikator]} (%)", "yaxis_title": None})
    fig.update_layout(**layout)
    return fig


def create_histogram(df_kab: pd.DataFrame, indikator: str, period_label: str, multi_year: bool = False) -> go.Figure:
    if multi_year and "tahun" in df_kab.columns:
        d = df_kab.copy()
        d["tahun"] = d["tahun"].astype(str)
        fig = px.histogram(
            d,
            x=indikator,
            color="tahun",
            nbins=14,
            barmode="overlay",
            opacity=0.65,
            title=f"Histogram {INDICATOR_LABELS[indikator]} - Periode {period_label} (per Tahun)",
            labels={indikator: f"{INDICATOR_LABELS[indikator]} (%)", "tahun": "Tahun"},
        )
    else:
        fig = px.histogram(
            df_kab,
            x=indikator,
            nbins=14,
            color_discrete_sequence=[COLOR_SCHEME[indikator]],
            title=f"Histogram {INDICATOR_LABELS[indikator]} - Tahun {period_label}",
            labels={indikator: f"{INDICATOR_LABELS[indikator]} (%)"},
        )
    layout = base_layout()
    layout.update({"yaxis_title": "Frekuensi"})
    fig.update_layout(**layout)
    return fig


def create_boxplot(df_kab: pd.DataFrame, indikator: str) -> go.Figure:
    fig = px.box(
        df_kab,
        y=indikator,
        points="all",
        color_discrete_sequence=[COLOR_SCHEME[indikator]],
        title=f"Boxplot {INDICATOR_LABELS[indikator]} (Deteksi Outlier)",
        labels={indikator: f"{INDICATOR_LABELS[indikator]} (%)"},
    )
    fig.update_layout(**base_layout())
    return fig


def create_line_kabupaten(df_kab: pd.DataFrame, indikator: str, regions: List[str]) -> go.Figure:
    d = df_kab[df_kab["kabupaten"].isin(regions)].sort_values(["kabupaten", "tahun"])
    fig = go.Figure()

    for region in regions:
        sub = d[d["kabupaten"] == region]
        if sub.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=sub["tahun"],
                y=sub[indikator],
                mode="lines+markers",
                name=region,
                line={"width": 2},
                marker={"size": 7},
                legendgroup=region,
            )
        )

        imp_col = f"{indikator}_imputed"
        if imp_col in sub.columns and sub[imp_col].fillna(False).any():
            seg_mask = _build_imputed_segment_mask(sub["tahun"], sub[imp_col].fillna(False))
            seg = sub.copy()
            seg.loc[~seg_mask, indikator] = np.nan
            seg_symbol = ["diamond" if bool(v) else "circle-open" for v in seg[imp_col].fillna(False)]
            fig.add_trace(
                go.Scatter(
                    x=seg["tahun"],
                    y=seg[indikator],
                    mode="lines+markers",
                    name=f"{region} — {IMPUTED_LABEL}",
                    line={"width": 2.5, "dash": "dash"},
                    marker={"size": 8, "symbol": seg_symbol},
                    legendgroup=f"{region}_imputed",
                )
            )

    fig.update_layout(
        title=f"Time Series {INDICATOR_LABELS[indikator]} per Kabupaten/Kota",
        xaxis_title="Tahun",
        yaxis_title=f"{INDICATOR_LABELS[indikator]} (%)",
    )
    fig.update_layout(**base_layout())
    return fig


def create_facet(df_kab: pd.DataFrame, indikator: str, cols: int = 5) -> go.Figure:
    """Single indikator di semua wilayah (small multiples klasik)"""
    # Susun urutan wilayah berdasarkan rata-rata indikator tertinggi.
    order = (
        df_kab.groupby("kabupaten", observed=False)[indikator]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # Hitung jumlah baris subplot berdasarkan jumlah wilayah.
    rows = (len(order) + cols - 1) // cols

    # Cari batas nilai global agar semua panel memakai skala Y yang sama.
    y_min = float(df_kab[indikator].min())
    y_max = float(df_kab[indikator].max())
    y_span = max(1e-9, y_max - y_min)
    y_pad = 0.08 * y_span
    y_range = [y_min - y_pad, y_max + y_pad]

    # Buat grid subplot kecil untuk setiap kabupaten.
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=order,
        specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)],
    )

    # Gambar satu garis per wilayah pada panel masing-masing.
    for i, region in enumerate(order):
        # Tentukan koordinat subplot saat ini.
        r = (i // cols) + 1
        c = (i % cols) + 1
        # Ambil data wilayah yang sedang diproses.
        sub = df_kab[df_kab["kabupaten"] == region]
        # Tambahkan line chart wilayah ke panel subplot.
        fig.add_trace(
            go.Scatter(
                x=sub["tahun"],
                y=sub[indikator],
                mode="lines+markers",
                name=region,
                line={"color": COLOR_SCHEME[indikator], "width": 2},
                marker={"size": 5},
            ),
            row=r,
            col=c,
        )

        imp_col = f"{indikator}_imputed"
        if imp_col in sub.columns and sub[imp_col].fillna(False).any():
            seg_mask = _build_imputed_segment_mask(sub["tahun"], sub[imp_col].fillna(False))
            seg = sub.copy()
            seg.loc[~seg_mask, indikator] = np.nan
            seg_symbol = ["diamond" if bool(v) else "circle-open" for v in seg[imp_col].fillna(False)]
            fig.add_trace(
                go.Scatter(
                    x=seg["tahun"],
                    y=seg[indikator],
                    mode="lines+markers",
                    name=IMPUTED_LABEL,
                    line={"color": COLOR_SCHEME[indikator], "width": 2.4, "dash": "dash"},
                    marker={"size": 6, "symbol": seg_symbol},
                    showlegend=(i == 0),
                    legendgroup="imputed_marker",
                ),
                row=r,
                col=c,
            )

    # Terapkan layout dasar agar konsisten dengan chart lain.
    layout = base_layout()
    # Judul dan tinggi chart disesuaikan dengan jumlah panel.
    layout.update({"title_text": f"Small Multiples {INDICATOR_LABELS[indikator]} (1 panel = 1 wilayah)", "height": 180 * rows, "showlegend": False})
    # Terapkan layout ke figure.
    fig.update_layout(**layout)
    # Paksa semua panel memakai skala Y yang sama.
    fig.update_yaxes(range=y_range)
    # Kembalikan figure siap tampil.
    return fig


def create_small_multiples_all_indicators(df_kab: pd.DataFrame, cols: int = 5) -> go.Figure:
    """Small multiples: semua kabupaten (10 panel), setiap panel punya 3 line (stunting, underweight, wasting)"""
    # Urutan panel berdasarkan rata-rata gabungan ketiga indikator.
    order = (
        df_kab.groupby("kabupaten", observed=False)[INDICATORS]
        .mean()
        .mean(axis=1)
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # Hitung jumlah baris subplot yang dibutuhkan.
    rows = (len(order) + cols - 1) // cols

    # Cari batas global seluruh indikator agar skala Y konsisten.
    y_min = float(df_kab[INDICATORS].min().min())
    y_max = float(df_kab[INDICATORS].max().max())
    y_span = max(1e-9, y_max - y_min)
    y_pad = 0.08 * y_span
    y_range = [y_min - y_pad, y_max + y_pad]

    # Bangun grid subplot untuk semua kabupaten.
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=order,
        specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)],
    )

    # Untuk setiap wilayah, gambar tiga garis indikator.
    for i, region in enumerate(order):
        # Cari posisi subplot.
        r = (i // cols) + 1
        c = (i % cols) + 1
        # Ambil data wilayah spesifik.
        sub = df_kab[df_kab["kabupaten"] == region]
        # Tambahkan tiga line indikator pada panel wilayah ini.
        for ind in INDICATORS:
            fig.add_trace(
                go.Scatter(
                    x=sub["tahun"],
                    y=sub[ind],
                    mode="lines+markers",
                    name=INDICATOR_LABELS[ind],
                    line={"color": COLOR_SCHEME[ind], "width": 2},
                    marker={"size": 4},
                    legendgroup=ind,
                    showlegend=(i == 0),  # Legend hanya muncul di panel pertama
                ),
                row=r,
                col=c,
            )

            imp_col = f"{ind}_imputed"
            if imp_col in sub.columns and sub[imp_col].fillna(False).any():
                seg_mask = _build_imputed_segment_mask(sub["tahun"], sub[imp_col].fillna(False))
                seg = sub.copy()
                seg.loc[~seg_mask, ind] = np.nan
                seg_symbol = ["diamond" if bool(v) else "circle-open" for v in seg[imp_col].fillna(False)]
                fig.add_trace(
                    go.Scatter(
                        x=seg["tahun"],
                        y=seg[ind],
                        mode="lines+markers",
                        name=f"{INDICATOR_LABELS[ind]} — {IMPUTED_LABEL}",
                        line={"color": COLOR_SCHEME[ind], "width": 2.4, "dash": "dash"},
                        marker={"size": 5, "symbol": seg_symbol},
                        legendgroup=f"{ind}_imputed",
                        showlegend=(i == 0),
                    ),
                    row=r,
                    col=c,
                )

    # Beri label sumbu pada baris paling bawah.
    fig.update_xaxes(title_text="Tahun", row=rows, col=1)
    fig.update_yaxes(title_text="(%)", row=rows, col=1)

    # Terapkan layout dasar dan set tinggi chart.
    layout = base_layout()
    layout.update({
        "title_text": "Small Multiples: Semua Kabupaten (1 panel = 1 wilayah; 3 garis = 3 indikator)",
        "height": 200 * rows,
        "showlegend": True,
    })
    # Terapkan layout.
    fig.update_layout(**layout)
    # Paksa semua subplot memakai range Y yang sama.
    fig.update_yaxes(range=y_range)
    # Kembalikan figure.
    return fig


def create_heatmap(df_kab: pd.DataFrame, indikator: str) -> go.Figure:
    # Bentuk matriks tahun x kabupaten untuk heatmap.
    pivot = df_kab.pivot_table(values=indikator, index="kabupaten", columns="tahun", aggfunc="first")
    # Buat visual heatmap dari matriks pivot.
    fig = px.imshow(
        pivot,
        labels={"x": "Tahun", "y": "Kabupaten/Kota", "color": f"{INDICATOR_LABELS[indikator]} (%)"},
        color_continuous_scale=COLOR_SCALE[indikator],
        aspect="auto",
        title=f"Heatmap {INDICATOR_LABELS[indikator]}",
    )
    # Tambahkan tinggi chart agar mudah dibaca.
    layout = base_layout()
    layout.update({"height": 500})
    # Terapkan layout ke chart.
    fig.update_layout(**layout)
    # Kembalikan heatmap.
    return fig


def create_indicator_compare(df_prov: pd.DataFrame) -> go.Figure:
    # Siapkan figure kosong untuk tiga indikator.
    fig = go.Figure()
    # Tambahkan satu garis per indikator.
    for ind in INDICATORS:
        fig.add_trace(
            go.Scatter(
                x=df_prov["tahun"],
                y=df_prov[ind],
                mode="lines+markers",
                name=INDICATOR_LABELS[ind],
                line={"color": COLOR_SCHEME[ind], "width": 2.5},
                marker={"size": 9},
            )
        )
    # Atur judul dan label sumbu agar konsisten.
    layout = base_layout()
    layout.update({"title": "Perbandingan Antar Indikator (Provinsi)", "xaxis_title": "Tahun", "yaxis_title": "Persentase (%)"})
    # Terapkan layout.
    fig.update_layout(**layout)
    # Kembalikan chart perbandingan.
    return fig


def create_scatter_with_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    color_col: str = "tahun",
    imputed_col: Optional[str] = None,
) -> go.Figure:
    # Ambil kolom yang dibutuhkan dan buang nilai kosong.
    d = df[[x_col, y_col, color_col, "kabupaten"]].dropna().copy()

    # Buat scatter plot utama.
    if imputed_col is not None and imputed_col in df.columns:
        d[imputed_col] = df.loc[d.index, imputed_col].fillna(False).astype(bool)
        fig = px.scatter(
            d,
            x=x_col,
            y=y_col,
            color=color_col,
            symbol=imputed_col,
            symbol_map={False: "circle", True: "diamond"},
            hover_name="kabupaten",
            title=title,
            labels={x_col: x_label, y_col: y_label, color_col: "Tahun", imputed_col: IMPUTED_LABEL},
            opacity=0.85,
        )

        # Rapikan label legend simbol imputasi agar lebih jelas.
        for tr in fig.data:
            if str(tr.name) == "True":
                tr.name = IMPUTED_LABEL
            elif str(tr.name) == "False":
                tr.name = "Observed"
    else:
        fig = px.scatter(
            d,
            x=x_col,
            y=y_col,
            color=color_col,
            hover_name="kabupaten",
            title=title,
            labels={x_col: x_label, y_col: y_label, color_col: "Tahun"},
            opacity=0.85,
        )

    # Tambahkan garis regresi linear sederhana jika datanya cukup.
    if len(d) >= 2:
        # Konversi sumbu X dan Y ke array numerik.
        x = d[x_col].to_numpy(dtype=float)
        y = d[y_col].to_numpy(dtype=float)
        # Hitung slope dan intercept garis lurus.
        b, a = np.polyfit(x, y, 1)
        # Bangun rentang X untuk garis regresi.
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        # Hitung nilai Y hasil garis regresi.
        y_line = b * x_line + a
        # Tambahkan garis regresi ke figure.
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Garis Regresi",
                line={"color": "#222222", "width": 2, "dash": "dash"},
            )
        )

    # Gunakan hover mode yang lebih presisi untuk scatter.
    layout = base_layout()
    layout.update({"hovermode": "closest"})
    # Terapkan layout.
    fig.update_layout(**layout)

    # Hindari tumpang tindih legend dengan colorbar tahun.
    fig.update_layout(
        legend={
            "x": 0.01,
            "y": 0.99,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "rgba(0,0,0,0.2)",
            "borderwidth": 1,
        }
    )

    if hasattr(fig.layout, "coloraxis") and fig.layout.coloraxis is not None:
        fig.update_layout(
            coloraxis_colorbar={
                "title": "Tahun",
                "x": 1.02,
                "y": 0.5,
                "len": 0.82,
                "thickness": 16,
            }
        )

    # Kembalikan chart scatter.
    return fig


def create_correlation_heatmap(df: pd.DataFrame, indikator: str) -> go.Figure:
    # Susun kolom yang akan dikorelasikan.
    cols = [indikator, "persen_kemiskinan", "akses_sanitasi", "ipm", "pelayanan_kesehatan"]
    # Buang data kosong supaya korelasi valid.
    d = df[cols].dropna().copy()
    # Hitung korelasi Pearson; kalau kosong, buat matriks NaN.
    corr = d.corr(method="pearson") if not d.empty else pd.DataFrame(np.nan, index=cols, columns=cols)

    # Ganti nama kolom agar lebih ramah di visual.
    rename_map = {
        indikator: INDICATOR_LABELS[indikator],
        "persen_kemiskinan": "Kemiskinan (%)",
        "akses_sanitasi": "Akses Sanitasi (%)",
        "ipm": "IPM",
        "pelayanan_kesehatan": "Pelayanan Kesehatan (%)",
    }
    # Terapkan label baru ke baris dan kolom.
    corr = corr.rename(index=rename_map, columns=rename_map)

    # Buat heatmap dari matriks korelasi.
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"Heatmap Korelasi {INDICATOR_LABELS[indikator]} (Pearson)",
    )
    # Set tinggi chart supaya angka terlihat jelas.
    layout = base_layout()
    layout.update({"height": 420})
    # Terapkan layout.
    fig.update_layout(**layout)
    # Kembalikan heatmap korelasi.
    return fig


def correlation_strength_label(r: float) -> str:
    """Kategori kekuatan korelasi berdasarkan |r|."""
    a = abs(r)
    if a < 0.20:
        return "sangat lemah"
    if a < 0.40:
        return "lemah"
    if a < 0.60:
        return "sedang"
    if a < 0.80:
        return "kuat"
    return "sangat kuat"


def correlation_direction_label(r: float) -> str:
    if r > 0:
        return "positif (searah)"
    if r < 0:
        return "negatif (berlawanan arah)"
    return "netral"


def build_correlation_interpretation(x_name: str, y_name: str, r: float) -> str:
    strength = correlation_strength_label(r)
    direction = correlation_direction_label(r)
    return (
        f"**{x_name} vs {y_name}**: $r={r:.3f}$ → hubungan **{direction}** dengan kekuatan **{strength}**."
    )


def create_change_bar(df_kab: pd.DataFrame, indikator: str, y0: Optional[int] = None, y1: Optional[int] = None) -> Tuple[go.Figure, pd.DataFrame]:
    # Ambil tahun awal otomatis jika belum diberikan.
    y0 = min(df_kab["tahun"].dropna().astype(int).unique()) if y0 is None else y0
    # Ambil tahun akhir otomatis jika belum diberikan.
    y1 = max(df_kab["tahun"].dropna().astype(int).unique()) if y1 is None else y1
    # Ambil nilai awal per kabupaten.
    start = df_kab[df_kab["tahun"] == y0][["kabupaten", indikator]].rename(columns={indikator: "awal"})
    # Ambil nilai akhir per kabupaten.
    end = df_kab[df_kab["tahun"] == y1][["kabupaten", indikator]].rename(columns={indikator: "akhir"})

    # Gabungkan nilai awal dan akhir.
    d = start.merge(end, on="kabupaten")
    # Hitung perubahan antar tahun.
    d["perubahan"] = d["akhir"] - d["awal"]
    # Urutkan dari perubahan paling kecil ke besar.
    d = d.sort_values("perubahan")

    # Warna hijau untuk turun, merah untuk naik.
    colors = ["#00CC96" if v < 0 else "#EF553B" for v in d["perubahan"]]

    # Buat bar chart horizontal perubahan.
    fig = px.bar(
        d,
        x="perubahan",
        y="kabupaten",
        orientation="h",
        text="perubahan",
        title=f"Analisis Perubahan {INDICATOR_LABELS[indikator]} ({y0}-{y1})",
        labels={"perubahan": "Perubahan (%)", "kabupaten": "Kabupaten/Kota"},
    )
    # Tampilkan nilai perubahan pada ujung bar.
    fig.update_traces(marker_color=colors, texttemplate="%{x:.2f}%", textposition="outside")

    # Layout dasar agar chart tetap konsisten.
    layout = base_layout()
    layout.update({"showlegend": False})
    # Terapkan layout.
    fig.update_layout(**layout)
    # Kembalikan figure beserta tabel perubahan.
    return fig, d


def kpi_stats(df_kab: pd.DataFrame, df_prov: pd.DataFrame, indikator: str, tahun: Optional[int] = None) -> Dict[str, float]:
    # Jika tahun diberikan dan kolom tahun masih numerik, filter ke tahun itu.
    if tahun is not None and ("tahun" in df_kab.columns) and pd.api.types.is_numeric_dtype(df_kab["tahun"]):
        d = df_kab[df_kab["tahun"] == tahun]
    else:
        # Kalau tidak, pakai seluruh periode yang sudah dipilih.
        d = df_kab.copy()

    # Terapkan logika serupa untuk data provinsi.
    if tahun is not None and ("tahun" in df_prov.columns) and pd.api.types.is_numeric_dtype(df_prov["tahun"]):
        p = df_prov[df_prov["tahun"] == tahun]
    else:
        # Kalau agregasi periode, rata-rata provinsi dipakai.
        p = df_prov.copy()

    # Kembalikan nilai KPI yang dibutuhkan di header.
    return {
        "rata2": float(d[indikator].mean()),
        "prov": float(p[indikator].mean()) if not p.empty else float("nan"),
        "max": float(d[indikator].max()),
        "min": float(d[indikator].min()),
        "kab_max": str(d.loc[d[indikator].idxmax(), "kabupaten"]),
        "kab_min": str(d.loc[d[indikator].idxmin(), "kabupaten"]),
    }


def resolve_year_selection(selected: List, all_years: List[int]) -> List[int]:
    """Resolve multiselect values with optional (All) sentinel."""
    # Jika kosong atau memilih (All), kembalikan semua tahun.
    if not selected or "(All)" in selected:
        return list(all_years)
    # Jika tidak, rapikan urutan dan pastikan bertipe integer.
    return sorted({int(y) for y in selected})


def format_year_selection(years: List[int]) -> str:
    """Format tahun terpilih menjadi rentang kompak.

    Contoh:
    - [2021, 2022, 2023, 2024] -> 2021-2024
    - [2019, 2021, 2024] -> 2019, 2021, 2024
    - [2019, 2021, 2022, 2023] -> 2019, 2021-2023
    """
    # Pastikan tahun unik dan terurut.
    ys = sorted({int(y) for y in years})
    if not ys:
        # Jika kosong, kembalikan string kosong.
        return ""
    if len(ys) == 1:
        # Jika hanya satu tahun, tampilkan apa adanya.
        return str(ys[0])

    # Kumpulkan tahun berurutan ke dalam rentang.
    ranges = []
    start = prev = ys[0]
    for year in ys[1:]:
        # Jika masih berurutan, lanjutkan rentang saat ini.
        if year == prev + 1:
            prev = year
            continue
        # Kalau terputus, simpan rentang lama dan mulai yang baru.
        ranges.append((start, prev))
        start = prev = year
    # Simpan rentang terakhir.
    ranges.append((start, prev))

    # Format hasil menjadi teks rentang kompak.
    parts = [f"{a}-{b}" if a != b else str(a) for a, b in ranges]
    return ", ".join(parts)


def is_aggregated_period(df: pd.DataFrame) -> bool:
    # Jika kolom tahun bukan numerik, berarti sudah berupa periode agregasi.
    return "tahun" not in df.columns or not pd.api.types.is_numeric_dtype(df["tahun"])


def summarize_multi_year_kab(df_kab: pd.DataFrame, selected_years: List[int], period_label: str) -> pd.DataFrame:
    """Rata-rata per kabupaten untuk periode terpilih."""
    # Ambil hanya baris pada tahun yang dipilih.
    d = df_kab[df_kab["tahun"].isin(selected_years)].copy()
    if len(selected_years) <= 1:
        # Kalau hanya satu tahun, tidak perlu agregasi.
        return d

    # Ambil semua kolom numerik selain tahun.
    numeric_cols = [c for c in d.select_dtypes(include=[np.number]).columns.tolist() if c != "tahun"]
    # Hitung rata-rata per kabupaten untuk seluruh indikator numerik.
    summary = d.groupby("kabupaten", as_index=False)[numeric_cols].mean()
    # Simpan label periode agar chart tahu ini hasil agregasi.
    summary["tahun"] = period_label
    # Kembalikan data ringkasan periode.
    return summary


def summarize_multi_year_prov(df_prov: pd.DataFrame, selected_years: List[int], period_label: str) -> pd.DataFrame:
    """Rata-rata provinsi untuk periode terpilih."""
    # Ambil baris provinsi pada tahun yang dipilih.
    d = df_prov[df_prov["tahun"].isin(selected_years)].copy()
    if len(selected_years) <= 1:
        # Kalau cuma satu tahun, pakai data asli.
        return d

    # Pilih kolom numerik untuk dihitung rata-ratanya.
    numeric_cols = [c for c in d.select_dtypes(include=[np.number]).columns.tolist() if c != "tahun"]
    # Buat satu baris ringkasan periode.
    summary = d[numeric_cols].mean().to_frame().T
    # Tambahkan label periode pada hasil ringkasan.
    summary["tahun"] = period_label
    # Susun kolom agar tetap rapi.
    return summary[["tahun"] + numeric_cols]


# =============================================================================
# WEB SECTION RENDERERS (A - L)
# =============================================================================


def render_sidebar(df_kab: pd.DataFrame, region_whitelist: List[str]) -> Tuple[int, str]:
    """Legacy function - kept for reference but no longer used in main()"""
    # Header kecil di sidebar untuk penanda area pengaturan.
    st.sidebar.markdown("## Settings")
    # Pilih tahun tunggal pada versi lama fungsi ini.
    tahun = st.sidebar.selectbox("Pilih Tahun", YEARS, index=len(YEARS) - 1)
    # Pilih indikator pada versi lama fungsi ini.
    indikator = st.sidebar.selectbox("Pilih Indikator", INDICATORS, format_func=lambda x: INDICATOR_LABELS[x])
    # Kembalikan tahun dan indikator yang dipilih.
    return tahun, indikator


def section_header(debug_id: str, title: str, subtitle: Optional[str] = None) -> None:
    """Legacy function - kept for reference but no longer used in main()"""
    # Tampilkan judul section dengan label debug.
    st.markdown(f"## {debug_id} {title}")
    # Jika ada subtitle, tampilkan sebagai caption.
    if subtitle:
        st.caption(subtitle)


def main() -> None:
    # load data sekali
    df_kab, df_prov, geojson, region_whitelist = load_all()
    
    st.sidebar.markdown("## ⚙️ Pengaturan")
    year_choices_raw = st.sidebar.multiselect("Pilih Tahun", ["(All)"] + YEARS, default=["(All)"])
    year_choices = resolve_year_selection(year_choices_raw, YEARS)
    indikator = st.sidebar.selectbox("Pilih Indikator", INDICATORS, format_func=lambda x: INDICATOR_LABELS[x])

    if not year_choices:
        year_choices = [max(YEARS)]
    active_year = max(year_choices)
    is_multi_year = len(year_choices) > 1
    period_label = format_year_selection(year_choices)

    kab_period = df_kab[df_kab["tahun"].isin(year_choices)].copy()
    prov_period = df_prov[df_prov["tahun"].isin(year_choices)].copy()
    kab_main = summarize_multi_year_kab(df_kab, year_choices, period_label) if is_multi_year else kab_period.copy()
    prov_main = summarize_multi_year_prov(df_prov, year_choices, period_label) if is_multi_year else prov_period.copy()

    st.markdown(f"# 📊 Dashboard Analisis Data Gizi NTB — {period_label}")
    st.caption("Jika memilih lebih dari 1 tahun, visual non-temporal memakai rata-rata periode terpilih.")

    st.sidebar.markdown("---")
    st.sidebar.info(INDICATOR_DESCRIPTIONS[indikator])
    st.sidebar.markdown(
        f"""
**Info Data**
- Tahun tersedia: {YEARS}
- Tahun aktif: **{period_label}**
- Wilayah resmi NTB: **{len(region_whitelist)}**
- Wilayah data aktif: **{df_kab['kabupaten'].nunique()}**
- Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    )

    missing = [r for r in OFFICIAL_NTB_REGIONS if r not in sorted(df_kab["kabupaten"].unique())]
    if missing:
        st.sidebar.warning("Wilayah belum ditemukan di data: " + ", ".join(missing))

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview",
        "📈 Trend & Perbandingan",
        "📉 Time Series",
        "🔥 Heatmap",
        "📊 Analisis Perubahan",
        "🔗 Korelasi",
        "💡 Insights",
        "📚 Sumber Data",
    ])

    # =========================================================================
    # TAB 1: OVERVIEW (KPI + Choropleth Map)
    # =========================================================================
    with tab1:
        st.markdown("## KPI Metrics")
        s = kpi_stats(kab_main, prov_main, indikator, None if is_multi_year else active_year)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rata-rata Kabupaten", f"{s['rata2']:.2f}%")
        c2.metric("Nilai Provinsi", f"{s['prov']:.2f}%")
        c3.metric("Maksimum", f"{s['max']:.2f}%", delta=s["kab_max"], delta_color="off")
        c4.metric("Minimum", f"{s['min']:.2f}%", delta=s["kab_min"], delta_color="off")

        st.markdown("---")
        st.markdown("## Choropleth Map - Analisis Spasial NTB")
        fig_c = create_choropleth(kab_main, geojson, indikator, period_label, multi_year=is_multi_year)
        st.plotly_chart(fig_c, width="stretch", key="choropleth_map")

    # =========================================================================
    # TAB 2: TREND PROVINSI + PERBANDINGAN KABUPATEN
    # =========================================================================
    with tab2:
        st.markdown("## Tren {0} di Provinsi NTB".format(INDICATOR_LABELS[indikator]))
        fig_d = create_line_provinsi(prov_period)
        st.plotly_chart(fig_d, width="stretch", key="overview_provinsi")

        st.markdown("## Ranking Kabupaten/Kota")
        top_n = st.slider("Top N wilayah", min_value=3, max_value=max(3, len(region_whitelist)), value=min(10, len(region_whitelist)))
        fig_e = create_bar_kabupaten(kab_main, indikator, period_label, top_n)
        st.plotly_chart(fig_e, width="stretch", key="comparison_kabupaten")

        st.markdown("---")
        st.markdown("## Distribusi Data (EDA)")
        f1, f2 = st.columns(2)
        with f1:
            st.plotly_chart(
                create_histogram(kab_period, indikator, period_label, multi_year=is_multi_year),
                width="stretch",
                key="histogram_eda",
            )
        with f2:
            st.plotly_chart(create_boxplot(kab_main, indikator), width="stretch", key="boxplot_eda")

    # =========================================================================
    # TAB 3: TIME SERIES (Selected Districts + Province)
    # =========================================================================
    with tab3:
        st.markdown("## Time Series Analysis")
        ts_sub_1, ts_sub_2 = st.tabs(["Line Time Series", "Small Multiples"])

        with ts_sub_1:
            all_regions = sorted(kab_period["kabupaten"].unique())
            selected_regions_raw = st.multiselect(
                "Pilih kabupaten/kota untuk ditampilkan",
                ["(All)"] + all_regions,
                default=["(All)"],
                key="ts_region_filter",
            )
            selected_regions = all_regions if (not selected_regions_raw or "(All)" in selected_regions_raw) else selected_regions_raw

            if selected_regions:
                st.plotly_chart(create_line_kabupaten(kab_period, indikator, selected_regions), width="stretch", key="timeseries_kabupaten")
            else:
                st.warning("Silakan pilih minimal 1 kabupaten/kota")

            st.markdown("---")
            st.markdown("## Tren Provinsi (All Indicators)")
            st.plotly_chart(create_line_provinsi(prov_period), width="stretch", key="timeseries_provinsi")

            st.caption("Catatan visual: segmen garis putus-putus + marker ◆ menandai data hasil imputasi 2020.")

        with ts_sub_2:
            st.markdown("## Perbandingan Semua Kabupaten/Kota")
            st.markdown("Setiap panel = 1 wilayah; 3 garis = stunting (🔴 merah), underweight (🟠 oranye), wasting (🟢 hijau)")
            sm_mode = st.radio("Tampilkan", ["Semua Indikator", "Indikator Terpilih"], horizontal=True)

            if sm_mode == "Semua Indikator":
                st.plotly_chart(create_small_multiples_all_indicators(kab_period, cols=5), width="stretch", key="small_multiples_all")
            else:
                st.plotly_chart(create_facet(kab_period, indikator, cols=5), width="stretch", key="small_multiples_single")

            st.caption("Catatan visual: segmen garis putus-putus + marker ◆ menandai data hasil imputasi 2020.")

    # =========================================================================
    # TAB 5: HEATMAP (Tahun x Kabupaten)
    # =========================================================================
    with tab4:
        st.markdown("## Heatmap Densitas")
        st.markdown("X = Tahun | Y = Kabupaten/Kota | Warna = Nilai Indikator")
        st.plotly_chart(create_heatmap(kab_period, indikator), width="stretch", key="heatmap_density")

    # =========================================================================
    # TAB 6: CHANGE ANALYSIS (Perubahan + Statistics)
    # =========================================================================
    with tab5:
        change_start = min(year_choices)
        change_end = max(year_choices)
        st.markdown("## Analisis Perubahan Indikator ({0} - {1})".format(change_start, change_end))
        fig_k, change_df = create_change_bar(kab_period, indikator, change_start, change_end)
        st.plotly_chart(fig_k, width="stretch", key="change_analysis")

        k1, k2 = st.columns(2)
        with k1:
            st.markdown("### 🟢 Top 3 Penurunan Terbaik")
            st.dataframe(change_df.head(3), width="stretch", hide_index=True)
        with k2:
            st.markdown("### 🔴 Top 3 Peningkatan Terbesar")
            positive_increase = change_df[change_df["perubahan"] > 0].sort_values("perubahan", ascending=False).head(3)
            if positive_increase.empty:
                st.info("Tidak ada wilayah dengan peningkatan positif pada rentang tahun terpilih.")
            else:
                st.dataframe(positive_increase, width="stretch", hide_index=True)

    # =========================================================================
    # TAB 7: KORELASI (Stunting vs Kemiskinan, Stunting vs Sanitasi)
    # =========================================================================
    with tab6:
        st.markdown(f"## Korelasi {INDICATOR_LABELS[indikator]} dengan Kemiskinan, Sanitasi, IPM, dan Pelayanan Kesehatan")
        st.caption("Metode: Pearson | Level analisis: kabupaten per tahun (panel data)")

        corr_year_choices_raw = st.multiselect(
            "Filter tahun korelasi (boleh pilih lebih dari 1)",
            options=["(All)"] + YEARS,
            default=["(All)"],
            key="corr_year_filter",
        )

        corr_year_choices = resolve_year_selection(corr_year_choices_raw, YEARS)

        if corr_year_choices:
            df_corr = df_kab[df_kab["tahun"].isin(corr_year_choices)].copy()
        else:
            df_corr = df_kab.iloc[0:0].copy()

        c1, c2 = st.columns(2)
        with c1:
            fig_corr_1 = create_scatter_with_regression(
                df_corr,
                x_col="persen_kemiskinan",
                y_col=indikator,
                title=f"{INDICATOR_LABELS[indikator]} vs Tingkat Kemiskinan",
                x_label="Tingkat Kemiskinan (%)",
                y_label=f"{INDICATOR_LABELS[indikator]} (%)",
                imputed_col=f"{indikator}_imputed",
            )
            st.plotly_chart(fig_corr_1, width="stretch", key="corr_stunting_kemiskinan")
        with c2:
            fig_corr_2 = create_scatter_with_regression(
                df_corr,
                x_col="akses_sanitasi",
                y_col=indikator,
                title=f"{INDICATOR_LABELS[indikator]} vs Akses Sanitasi",
                x_label="Akses Sanitasi (%)",
                y_label=f"{INDICATOR_LABELS[indikator]} (%)",
                imputed_col=f"{indikator}_imputed",
            )
            st.plotly_chart(fig_corr_2, width="stretch", key="corr_stunting_sanitasi")

        c3, c4 = st.columns(2)
        with c3:
            fig_corr_3 = create_scatter_with_regression(
                df_corr,
                x_col="ipm",
                y_col=indikator,
                title=f"{INDICATOR_LABELS[indikator]} vs IPM",
                x_label="IPM",
                y_label=f"{INDICATOR_LABELS[indikator]} (%)",
                imputed_col=f"{indikator}_imputed",
            )
            st.plotly_chart(fig_corr_3, width="stretch", key="corr_stunting_ipm")
        with c4:
            fig_corr_4 = create_scatter_with_regression(
                df_corr,
                x_col="pelayanan_kesehatan",
                y_col=indikator,
                title=f"{INDICATOR_LABELS[indikator]} vs Pelayanan Kesehatan Bayi",
                x_label="Pelayanan Kesehatan Bayi (%)",
                y_label=f"{INDICATOR_LABELS[indikator]} (%)",
                imputed_col=f"{indikator}_imputed",
            )
            st.plotly_chart(fig_corr_4, width="stretch", key="corr_stunting_pelkes")

        st.markdown("---")
        st.plotly_chart(create_correlation_heatmap(df_corr, indikator), width="stretch", key="corr_heatmap")

        corr_data = df_corr[[indikator, "persen_kemiskinan", "akses_sanitasi", "ipm", "pelayanan_kesehatan"]].dropna()
        if len(corr_data) >= 2:
            r_st_pov = corr_data[indikator].corr(corr_data["persen_kemiskinan"], method="pearson")
            r_st_san = corr_data[indikator].corr(corr_data["akses_sanitasi"], method="pearson")
            r_st_ipm = corr_data[indikator].corr(corr_data["ipm"], method="pearson")
            r_st_pel = corr_data[indikator].corr(corr_data["pelayanan_kesehatan"], method="pearson")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"r({INDICATOR_LABELS[indikator]}, Kemiskinan)", f"{r_st_pov:.3f}")
            m2.metric(f"r({INDICATOR_LABELS[indikator]}, Sanitasi)", f"{r_st_san:.3f}")
            m3.metric(f"r({INDICATOR_LABELS[indikator]}, IPM)", f"{r_st_ipm:.3f}")
            m4.metric(f"r({INDICATOR_LABELS[indikator]}, Pelayanan Kesehatan)", f"{r_st_pel:.3f}")

            st.markdown("### Interpretasi")
            st.markdown(
                "- " + build_correlation_interpretation("Kemiskinan", INDICATOR_LABELS[indikator], float(r_st_pov))
            )
            st.markdown(
                "- " + build_correlation_interpretation("Akses Sanitasi", INDICATOR_LABELS[indikator], float(r_st_san))
            )
            st.markdown(
                "- " + build_correlation_interpretation("IPM", INDICATOR_LABELS[indikator], float(r_st_ipm))
            )
            st.markdown(
                "- " + build_correlation_interpretation("Pelayanan Kesehatan", INDICATOR_LABELS[indikator], float(r_st_pel))
            )
            st.info(
                "Catatan: korelasi tidak sama dengan kausalitas; hasil ini menunjukkan pola hubungan, bukan sebab-akibat langsung."
            )
        else:
            st.warning("Data tidak cukup untuk menghitung korelasi pada filter tahun yang dipilih.")

    # =========================================================================
    # TAB 8: INSIGHTS (Ringkasan + Debug Data)
    # =========================================================================
    with tab7:
        st.markdown("## Ringkasan Insight")

        insight_df = kab_main.copy()
        w_max = insight_df.loc[insight_df[indikator].idxmax()]
        w_min = insight_df.loc[insight_df[indikator].idxmin()]

        first_year, last_year = min(year_choices), max(year_choices)
        avg_first = kab_period[kab_period["tahun"] == first_year][indikator].mean()
        avg_last = kab_period[kab_period["tahun"] == last_year][indikator].mean()
        delta = avg_last - avg_first

        st.markdown(
            f"""
**📊 Tren Utama:**
- Rata-rata {INDICATOR_LABELS[indikator]} berubah dari **{avg_first:.2f}%** ({first_year}) menjadi **{avg_last:.2f}%** ({last_year})
- Perubahan total: **{delta:+.2f} poin**

**🎯 Kondisi Periode {period_label}:**
- **Wilayah Tertinggi:** {w_max['kabupaten']} ({w_max[indikator]:.2f}%)
- **Wilayah Terendah:** {w_min['kabupaten']} ({w_min[indikator]:.2f}%)
- **Median:** {kab_main[indikator].median():.2f}%
- **Rentang (Range):** {kab_main[indikator].min():.2f}% - {kab_main[indikator].max():.2f}%

**💡 Tips Eksplorasi:**
- Lihat **Tab Overview** untuk visualisasi spasial (peta) dan KPI ringkas
- Lihat **Tab Trend & Perbandingan** untuk melihat tren provinsi sekaligus ranking kabupaten
- Lihat **Tab Time Series** (subtab Line + Small Multiples) untuk membandingkan pola antar wilayah
- Lihat **Tab Analisis Perubahan** untuk melihat wilayah dengan perbaikan terbaik
- Lihat **Tab Korelasi** untuk melihat relasi indikator dengan kemiskinan, sanitasi, IPM, dan pelayanan kesehatan
- Lihat **Tab Sumber Data** untuk referensi dataset dan catatan metodologi imputasi
"""
        )

        st.markdown("---")
        st.markdown("## Data Lengkap (Preview)")
        st.dataframe(kab_period.sort_values(["kabupaten", "tahun"]), width="stretch", hide_index=True)

    # =========================================================================
    # TAB 8: SUMBER DATA
    # =========================================================================
    with tab8:
        st.markdown("## Referensi Sumber Data")
        st.caption("Daftar sumber resmi yang dipakai pada dashboard ini.")

        src_df = pd.DataFrame(DATA_SOURCES)

        m1, m2 = st.columns(2)
        m1.metric("Total Dataset Sumber", f"{len(src_df)}")
        m2.metric("Instansi Sumber", f"{src_df['instansi'].nunique()}")

        st.markdown("### Daftar Sumber per Instansi")
        grouped_sources = src_df.sort_values(["instansi", "dataset"]).groupby("instansi", sort=False)
        for instansi, group in grouped_sources:
            with st.expander(f"🏛️ {instansi} ({len(group)} dataset)", expanded=True):
                for _, row in group.iterrows():
                    st.markdown(
                        f"""
- **{row['dataset']}**
  - Cakupan: {row['cakupan']}
  - Link: [Buka sumber]({row['url']})
"""
                    )

        st.markdown("---")
        st.markdown("### Tabel Ringkas")
        st.dataframe(src_df, width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("### Catatan Metodologi Data")
        st.markdown(
            f"""
- Data indikator gizi tahun **2020** yang hilang diimputasi menggunakan metode **MICE (IterativeImputer)**
  dengan estimator **RandomForestRegressor**.
- Fitur prediktor untuk imputasi: **Akses Sanitasi, Tingkat Kemiskinan, IPM, Pelayanan Kesehatan**,
  ditambah fitur **tahun** dan **encoding kabupaten**.
- Nilai hasil imputasi dibatasi pada rentang **0-100%**.
- Jika MICE gagal pada subset tertentu, fallback menggunakan **linear interpolation** per wilayah.
- Pada chart temporal, data imputasi ditandai dengan **garis putus-putus** dan marker **◆ ({IMPUTED_LABEL})**.
"""
        )


if __name__ == "__main__":
    # Jalankan aplikasi saat file dieksekusi langsung.
    main()

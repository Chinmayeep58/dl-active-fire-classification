import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import streamlit as st
import zipfile
import kagglehub
import glob
import planetary_computer
import pystac_client

from rasterio.windows import from_bounds
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor, as_completed

input_type = st.radio(
    "Input Type",
    [
        "Shapefile",
        "CSV"
    ]
)

if input_type == "Shapefile":

    dataset_name = st.text_input(
        "Kaggle Dataset",
        value="chinmayeep2385/final-2024"
    )

else:

    dataset_name = st.text_input(
        "Kaggle Dataset",
        value="chinmayeep2385/firms-fire-dataset"
    )
CONFIGS = [
    {
        "name": "mm",
        "output_dir": "sent_mm_doubt",
        "file_prefix": "sent_mm_doubt",
        "start_date": "2023-03-01",
        "end_date": "2023-05-31",
        "cloud_cover": 10,
        "cloud_threshold": 50
    },
    {
        "name": "od",
        "output_dir": "sent_od_doubt",
        "file_prefix": "sent_od_doubt",
        "start_date": "2023-10-01",
        "end_date": "2023-12-31",
        "cloud_cover": 50,
        "cloud_threshold": 60
    }
]


if st.button("Download Dataset"):

    dataset_path = kagglehub.dataset_download(
        dataset_name
    )

    st.session_state["dataset_path"] = dataset_path

    if input_type == "Shapefile":

        shp_path = glob.glob(
            os.path.join(
                dataset_path,
                "**/*.shp"
            ),
            recursive=True
        )[0]

        gdf = gpd.read_file(
            shp_path
        )

        gdf = gdf.to_crs(
            "EPSG:4326"
        )

        gdf = gdf[
            gdf["remark"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "doubt"
        ]

        gdf = gdf.reset_index(
            drop=True
        )

        st.session_state["input_df"] = gdf

        st.session_state["input_type"] = "Shapefile"

        st.success(
            f"{len(gdf)} polygons loaded."
        )

    else:

        csv_path = glob.glob(
            os.path.join(
                dataset_path,
                "**/*.csv"
            ),
            recursive=True
        )[0]

        df = pd.read_csv(
            csv_path
        )

        cols = {
            c.lower(): c
            for c in df.columns
        }

        lat_col = cols["latitude"]
        lon_col = cols["longitude"]

        if "brightness" in cols:

            df = df[
                df[
                    cols["brightness"]
                ] > 330
            ]

        if "confidence" in cols:

            df = df[
                df[
                    cols["confidence"]
                ]
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "h",
                        "high",
                        "n"
                    ]
                )
            ]

        if "instrument" in cols:

            df = df[
                df[
                    cols["instrument"]
                ]
                .astype(str)
                .str.upper()
                == "VIIRS"
            ]

        df = (
            df[
                [
                    lat_col,
                    lon_col
                ]
            ]
            .rename(
                columns={
                    lat_col:
                    "latitude",

                    lon_col:
                    "longitude"
                }
            )
            .dropna()
            .reset_index(
                drop=True
            )
        )

        st.session_state[
            "input_df"
        ] = df

        st.session_state[
            "input_type"
        ] = "CSV"

        st.success(
            f"{len(df)} points loaded."
        )

MM_DIR = "sent_mm_doubt"
OD_DIR = "sent_od_doubt"

os.makedirs(
    MM_DIR,
    exist_ok=True
)

os.makedirs(
    OD_DIR,
    exist_ok=True
)

PATCH_SIZE = 64
PATCH_METERS = 640

BANDS = [
    "B04",
    "B03",
    "B02",
    "B08"
]

MAX_WORKERS = 8


catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# st.session_state["period"] = period
st.session_state["patch_size"] = PATCH_SIZE
# ============================================================
# PROGRESS TRACKING
# ============================================================

for cfg in CONFIGS:

    cfg["log_file"] = os.path.join(
        cfg["output_dir"],
        "progress.csv"
    )

# ============================================================
# PROCESS TILE
# ============================================================
def process_tile(
    idx,
    row,
    input_type,
    output_dir,
    file_prefix,
    start_date,
    end_date,
    cloud_cover,
    cloud_threshold
):

    try:

        # =====================================================
        # GET LAT/LON
        # =====================================================
        if input_type == "Shapefile":

            geom = row.geometry

            if geom is None or geom.is_empty:
                raise Exception("Invalid geometry")

            centroid = geom.centroid

            lon = centroid.x
            lat = centroid.y

            xmin_geom, ymin_geom, xmax_geom, ymax_geom = geom.bounds

        else:

            lat = float(row["latitude"])
            lon = float(row["longitude"])

            xmin_geom = lon - 0.01
            xmax_geom = lon + 0.01

            ymin_geom = lat - 0.01
            ymax_geom = lat + 0.01

        # =====================================================
        # OUTPUT FILE
        # =====================================================
        out_file = (
            f"{output_dir}/"
            f"{file_prefix}_{idx}.tif"
        )

        if os.path.exists(out_file):

            return {
                "index": idx,
                "status": "exists"
            }

        # =====================================================
        # SEARCH SENTINEL
        # =====================================================
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=[
                xmin_geom,
                ymin_geom,
                xmax_geom,
                ymax_geom
            ],
            datetime=f"{start_date}/{end_date}",
            query={
                "eo:cloud_cover": {
                    "lt": cloud_cover
                }
            }
        )

        items = list(search.items())

        if len(items) == 0:
            raise Exception("No Sentinel scene")

        items = sorted(
            items,
            key=lambda x: x.properties["eo:cloud_cover"]
        )

        valid_item = None

        # =====================================================
        # CLOUD FILTERING
        # =====================================================
        for candidate_item in items:

            candidate_item = planetary_computer.sign(
                candidate_item
            )

            if "SCL" not in candidate_item.assets:
                continue

            scl_ds = rasterio.open(
                candidate_item.assets["SCL"].href
            )

            transformer = Transformer.from_crs(
                "EPSG:4326",
                scl_ds.crs,
                always_xy=True
            )

            x, y = transformer.transform(
                lon,
                lat
            )

            half = PATCH_METERS / 2

            xmin = x - half
            xmax = x + half

            ymin = y - half
            ymax = y + half

            window = from_bounds(
                xmin,
                ymin,
                xmax,
                ymax,
                scl_ds.transform
            )

            window = (
                window
                .round_offsets()
                .round_lengths()
            )

            scl_arr = scl_ds.read(
                1,
                window=window,
                out_shape=(PATCH_SIZE, PATCH_SIZE),
                boundless=True,
                fill_value=0
            )

            cloud_pixels = np.isin(
                scl_arr,
                [3, 8, 9, 10]
            )

            cloud_percent = (
                cloud_pixels.sum()
                /
                cloud_pixels.size
            ) * 100

            if cloud_percent < cloud_threshold:

                valid_item = candidate_item
                break

        if valid_item is None:

            raise Exception(
                "No low-cloud patch found"
            )

        # =====================================================
        # LOAD BANDS
        # =====================================================
        datasets = {}

        for band in BANDS:

            datasets[band] = rasterio.open(
                valid_item.assets[band].href
            )

        transformer = Transformer.from_crs(
            "EPSG:4326",
            datasets["B04"].crs,
            always_xy=True
        )

        x, y = transformer.transform(
            lon,
            lat
        )

        half = PATCH_METERS / 2

        xmin = x - half
        xmax = x + half

        ymin = y - half
        ymax = y + half

        bands = []

        valid_transform = None

        # =====================================================
        # EXTRACT BANDS
        # =====================================================
        for band in BANDS:

            ds = datasets[band]

            window = from_bounds(
                xmin,
                ymin,
                xmax,
                ymax,
                ds.transform
            )

            window = (
                window
                .round_offsets()
                .round_lengths()
            )

            arr = ds.read(
                1,
                window=window,
                out_shape=(PATCH_SIZE, PATCH_SIZE),
                boundless=True,
                fill_value=0
            )

            arr = np.nan_to_num(arr)

            valid_pixels = arr[arr > 0]

            if len(valid_pixels) == 0:

                raise Exception(
                    "All black patch"
                )

            p2 = np.percentile(
                valid_pixels,
                2
            )

            p98 = np.percentile(
                valid_pixels,
                98
            )

            if p98 <= p2:

                p2 = 0
                p98 = 1000

            arr = (
                (arr - p2)
                /
                (p98 - p2)
            ) * 255

            arr = np.clip(
                arr,
                0,
                255
            ).astype(np.uint8)

            bands.append(arr)

            valid_transform = rasterio.windows.transform(
                window,
                ds.transform
            )

        img = np.stack(
            bands,
            axis=0
        )

        # =====================================================
        # SAVE TIFF
        # =====================================================
        with rasterio.open(
            out_file,
            "w",
            driver="GTiff",
            height=PATCH_SIZE,
            width=PATCH_SIZE,
            count=4,
            dtype=np.uint8,
            crs=ds.crs,
            transform=valid_transform
        ) as dst:

            dst.write(img)

        return {
            "index": idx,
            "status": "success",
            "file": out_file
        }

    except Exception as e:

        return {
            "index": idx,
            "status": "failed",
            "error": str(e)
        }
        
# ============================================================
# RUN PROCESSING
# ============================================================
# st.session_state["sentinel_output_dir"] = OUTPUT_DIR

# if period == "March-May":

#     st.session_state["mm_patch_dir"] = OUTPUT_DIR

# else:

#     st.session_state["od_patch_dir"] = OUTPUT_DIR

if (
    "input_df" in st.session_state
    and
    st.button(
        "Start Extraction"
    )
):

    for cfg in CONFIGS:

        st.subheader(
            f"Processing {cfg['name'].upper()}"
        )

        LOG_FILE = cfg["log_file"]

        processed = set()

        if os.path.exists(LOG_FILE):

            processed = set(
                pd.read_csv(LOG_FILE)["index"]
            )

        results = []

        progress_bar = st.progress(0)

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            futures = []

            for idx, row in st.session_state["input_df"].iterrows():

                if idx in processed:
                    continue

                futures.append(
                    executor.submit(
                        process_tile,
                        idx,
                        row,
                        st.session_state["input_type"],
                        cfg["output_dir"],
                        cfg["file_prefix"],
                        cfg["start_date"],
                        cfg["end_date"],
                        cfg["cloud_cover"],
                        cfg["cloud_threshold"]
                    )
                )

            total = len(futures)

            if total > 0:

                for i, future in enumerate(
                    as_completed(futures),
                    start=1
                ):

                    results.append(
                        future.result()
                    )

                    progress_bar.progress(
                        i / total
                    )

        new_log = pd.DataFrame(results)

        if os.path.exists(LOG_FILE):

            old_log = pd.read_csv(
                LOG_FILE
            )

            final_log = pd.concat(
                [old_log, new_log],
                ignore_index=True
            )

        else:

            final_log = new_log

        final_log = final_log.drop_duplicates(
            subset=["index"],
            keep="last"
        )

        final_log.to_csv(
            LOG_FILE,
            index=False
        )

        patch_map = {}

        success_rows = final_log[
            final_log["status"].isin(
                ["success", "exists"]
            )
        ]

        for _, r in success_rows.iterrows():

            idx = int(r["index"])

            patch_path = os.path.join(
                cfg["output_dir"],
                f"{cfg['file_prefix']}_{idx}.tif"
            )

            if os.path.exists(patch_path):

                patch_map[idx] = patch_path

        if cfg["name"] == "mm":

            st.session_state[
                "mm_patch_map"
            ] = patch_map

            st.session_state[
                "mm_patch_dir"
            ] = cfg["output_dir"]

        else:

            st.session_state[
                "od_patch_map"
            ] = patch_map

            st.session_state[
                "od_patch_dir"
            ] = cfg["output_dir"]

        ZIP_PATH = (
            f"{cfg['output_dir']}.zip"
        )

        with zipfile.ZipFile(
            ZIP_PATH,
            "w",
            zipfile.ZIP_DEFLATED
        ) as z:

            for file in os.listdir(
                cfg["output_dir"]
            ):

                if file.endswith(".tif"):

                    z.write(
                        os.path.join(
                            cfg["output_dir"],
                            file
                        ),
                        arcname=file
                    )

        with open(
            ZIP_PATH,
            "rb"
        ) as f:

            st.download_button(
                f"Download {cfg['name'].upper()} Dataset",
                f,
                file_name=os.path.basename(
                    ZIP_PATH
                ),
                mime="application/zip"
            )

        st.success(
            f"{cfg['name'].upper()} : "
            f"{len(success_rows)} patches"
        )
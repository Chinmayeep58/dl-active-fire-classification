import os
import gc
import json
import shutil
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import mercantile
import streamlit as st 

from tqdm import tqdm
from PIL import Image
from io import BytesIO

from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

from pyproj import Transformer

# ============================================================
# CONFIG
# ============================================================
import kagglehub

input_type = st.radio(
    "Input Type",
    [
        "Shapefile",
        "CSV"
    ]
)

if input_type=="Shapefile":
    dataset_name = st.text_input(
        "Kaggle Dataset",
        value="chinmayeep2385/final-2024"
    )
else:
    dataset_name = st.text_input(
        "Kaggle Dataset",
        value="chinmayeep2385/firms-fire-dataset"
    )

OUTPUT_DIR = "esri_doubt_dataset"

PATCH_DIR = os.path.join(
    OUTPUT_DIR,
    "patches"
)

st.session_state["esri_patch_dir"] = PATCH_DIR

LOG_FILE = os.path.join(
    OUTPUT_DIR,
    "progress.csv"
)

ZOOM = 17

PATCH_SIZE = 512
GROUND_SIZE = 512

MAX_CONCURRENT = 6
MAX_RETRIES = 8

# ============================================================
# CREATE FOLDERS
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PATCH_DIR, exist_ok=True)

# ============================================================
# LOAD SHAPEFILE
# ============================================================
import glob
import kagglehub



# if st.button(
#     "Download Dataset"
# ):

#     dataset_path = kagglehub.dataset_download(
#         dataset_name
#     )

#     st.session_state["dataset_path"] = dataset_path

#     st.success(
#         "Dataset downloaded."
#     )

#     shp_path = glob.glob(
#         os.path.join(
#             dataset_path,
#             "**/*.shp"
#         ),
#         recursive=True
#     )[0]

#     st.session_state["shp_path"] = shp_path

#     gdf = gpd.read_file(
#         shp_path
#     )

# # Convert CRS
#     gdf = gdf.to_crs("EPSG:4326")

#     gdf = gdf[
#         gdf["remark"]
#         .astype(str)
#         .str.strip()
#         .str.lower()
#         == "doubt"
#     ]
    
#     gdf = gdf.reset_index(drop=True)
    
#     gdf["filename"] = (
#         "esri_doubt_"
#         + gdf.index.astype(str)
#         + ".tif"
#     )
    
#     st.session_state["gdf"] = gdf
#     st.session_state["tile_gdf"] = gdf.copy()
#     st.session_state["shp_path"] = shp_path
#     st.session_state["dataset_path"] = dataset_path

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

        gdf = gdf.to_crs("EPSG:4326")

        gdf = gdf[
            gdf["remark"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "doubt"
        ]

        gdf = gdf.reset_index(drop=True)

        gdf["filename"] = (
            "esri_doubt_"
            + gdf.index.astype(str)
            + ".tif"
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

        df = pd.read_csv(csv_path)

        cols = {
            c.lower(): c
            for c in df.columns
        }
        
        lat_col = cols["latitude"]
        lon_col = cols["longitude"]
        
        # ------------------------------------------------
        # FIRMS filtering
        # ------------------------------------------------
        if "brightness" in cols:
            df = df[
                df[cols["brightness"]] > 330
            ]
        
        if "confidence" in cols:
        
            conf_col = cols["confidence"]
        
            df = df[
                df[conf_col]
                .astype(str)
                .str.lower()
                .isin(["h", "high", "n"])
            ]
        
        # Optional
        if "instrument" in cols:
        
            df = df[
                df[cols["instrument"]]
                .astype(str)
                .str.upper()
                == "VIIRS"
            ]
        
        df = (
            df[
                [lat_col, lon_col]
            ]
            .rename(
                columns={
                    lat_col: "latitude",
                    lon_col: "longitude"
                }
            )
            .dropna().reset_index(drop=True)
        )
        
        df["filename"] = (
            "esri_doubt_"
            + df.index.astype(str)
            + ".tif"
        )

        st.session_state["input_df"] = df
        st.session_state["input_type"] = "CSV"

        st.success(
            f"{len(df)} points loaded."
        )


# ============================================================
# LOAD EXISTING LOG
# ============================================================
processed = set()

if os.path.exists(LOG_FILE):

    processed = set(
        pd.read_csv(LOG_FILE)["index"].tolist()
    )

# ============================================================
# TILE FETCH
# ============================================================
async def fetch_tile(session, x, y, z):

    url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    for _ in range(MAX_RETRIES):

        try:

            async with session.get(url, timeout=30) as r:

                if r.status != 200:
                    continue

                data = await r.read()

                img = Image.open(
                    BytesIO(data)
                ).convert("RGB")

                return np.array(img)

        except:

            await asyncio.sleep(2)

    return None

# ============================================================
# PROCESS SINGLE POLYGON
# ============================================================
async def process(idx, row, session):

    try:

        if st.session_state["input_type"] == "Shapefile":
        
            geom = row.geometry
        
            if geom is None:
        
                return {
                    "index": idx,
                    "status": "no_geometry"
                }
        
            centroid = geom.centroid
        
            lon = centroid.x
            lat = centroid.y
        
        else:
        
            lat = float(
                row["latitude"]
            )
        
            lon = float(
                row["longitude"]
            )

        # =====================================================
        # TRANSFORMERS
        # =====================================================
        to3857 = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:3857",
            always_xy=True
        )

        to4326 = Transformer.from_crs(
            "EPSG:3857",
            "EPSG:4326",
            always_xy=True
        )

        x, y = to3857.transform(lon, lat)

        half = GROUND_SIZE / 2

        xmin = x - half
        ymin = y - half
        xmax = x + half
        ymax = y + half

        lon_min, lat_min = to4326.transform(xmin, ymin)
        lon_max, lat_max = to4326.transform(xmax, ymax)

        # =====================================================
        # OUTPUT FILE
        # =====================================================
        out_path = f"{PATCH_DIR}/esri_doubt_{idx}.tif"

        if os.path.exists(out_path):

            return {
                "index": idx,
                "status": "exists"
            }

        # =====================================================
        # REQUIRED TILES
        # =====================================================
        tiles = list(
            mercantile.tiles(
                lon_min,
                lat_min,
                lon_max,
                lat_max,
                [ZOOM]
            )
        )

        tasks = []

        for t in tiles:

            tasks.append(
                fetch_tile(
                    session,
                    t.x,
                    t.y,
                    t.z
                )
            )

        imgs = await asyncio.gather(*tasks)

        datasets = []

        # =====================================================
        # TILE -> TEMP DATASET
        # =====================================================
        for t, img in zip(tiles, imgs):

            if img is None:
                continue

            bounds = mercantile.bounds(t)

            transform = from_bounds(
                bounds.west,
                bounds.south,
                bounds.east,
                bounds.north,
                256,
                256
            )

            profile = {
                "driver": "GTiff",
                "height": 256,
                "width": 256,
                "count": 3,
                "dtype": rasterio.uint8,
                "crs": "EPSG:4326",
                "transform": transform
            }

            memfile = MemoryFile()

            dataset = memfile.open(**profile)

            dataset.write(img[:, :, 0], 1)
            dataset.write(img[:, :, 1], 2)
            dataset.write(img[:, :, 2], 3)

            datasets.append(dataset)

        if len(datasets) == 0:

            return {
                "index": idx,
                "status": "failed_tiles"
            }

        # =====================================================
        # MERGE
        # =====================================================
        mosaic, out_transform = merge(datasets)

        # =====================================================
        # EXACT 512m x 512m @ 1m resolution
        # =====================================================
        dst_transform = rasterio.transform.from_origin(
            xmin,
            ymax,
            1,
            1
        )

        fixed = np.zeros(
            (3, PATCH_SIZE, PATCH_SIZE),
            dtype=np.uint8
        )

        for i in range(3):

            reproject(
                source=mosaic[i],
                destination=fixed[i],
                src_transform=out_transform,
                src_crs="EPSG:4326",
                dst_transform=dst_transform,
                dst_crs="EPSG:3857",
                resampling=Resampling.nearest
            )

        # black_fraction = (
        #     np.all(fixed == 0, axis=0)
        # ).mean()
        
        # if black_fraction > 0.80:
        
        #     return {
        #         "index": idx,
        #         "status": "mostly_black",
        #         "black_fraction": float(
        #             black_fraction
        #         )
        #     }
        # =====================================================
        # SAVE TIFF
        # =====================================================
        profile = {
            "driver": "GTiff",
            "height": PATCH_SIZE,
            "width": PATCH_SIZE,
            "count": 3,
            "dtype": rasterio.uint8,
            "crs": "EPSG:3857",
            "transform": dst_transform,
            "compress": "lzw"
        }

        with rasterio.open(
            out_path,
            "w",
            **profile
        ) as dst:

            dst.write(fixed)

        # =====================================================
        # CLEANUP
        # =====================================================
        for ds in datasets:
            ds.close()

        del mosaic
        del fixed
        del datasets

        gc.collect()

        return {
            "index": idx,
            "status": "saved",
            "tif_path": out_path,
            "latitude": lat,
            "longitude": lon
        }

    except Exception as e:

        return {
            "index": idx,
            "status": "failed",
            "error": str(e)
        }

# ============================================================
# MAIN
# ============================================================
async def main():

    results = []

    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT
    )

    async with aiohttp.ClientSession(
        connector=connector
    ) as session:

        tasks = []

        for idx, row in st.session_state["input_df"].iterrows():

            if idx in processed:
                continue

            tasks.append(
                process(idx, row, session)
            )

        progress_bar = st.progress(0)

        for i, f in enumerate(
            asyncio.as_completed(tasks)
        ):

            r = await f

            progress_bar.progress(
                (i+1)/len(tasks)
            )

            results.append(r)

    return results

# ============================================================
# RUN IN KAGGLE NOTEBOOK
# ============================================================
if (
    "input_df" in st.session_state
    and
    st.button(
        "Start Extraction"
    )
):

    results = asyncio.run(
        main()
    )

# ============================================================
# SAVE LOG
# ============================================================
    new_log = pd.DataFrame(results)

    if os.path.exists(LOG_FILE):
    
        old_log = pd.read_csv(LOG_FILE)
    
        final_log = pd.concat(
            [old_log, new_log],
            ignore_index=True
        )
    
    else:
    
        final_log = new_log
    
    final_log.to_csv(LOG_FILE, index=False)
    st.session_state["extraction_log"] = final_log
    success_rows = final_log[
        final_log["status"].isin(
            ["saved", "exists"]
        )
    ]
    
    esri_patch_map = {}
    
    for _, row in success_rows.iterrows():
    
        idx = int(row["index"])
    
        file_path = os.path.join(
            PATCH_DIR,
            f"esri_doubt_{idx}.tif"
        )
    
        if os.path.exists(file_path):
    
            esri_patch_map[idx] = file_path
    
    st.session_state["esri_patch_map"] = esri_patch_map

    # ============================================================
    # CREATE UNIFIED INPUT DF FOR PREDICTION PIPELINE
    # ============================================================
    
    success_rows = final_log[
        final_log["status"].isin(["saved", "exists"])
    ].copy()
    
    records = []

    for _, row in success_rows.iterrows():
    
        idx = int(row["index"])
        file_path = os.path.join(
            PATCH_DIR,
            f"esri_doubt_{idx}.tif"
        )
    
        if os.path.exists(file_path):
    
            records.append({
                "index": idx,
                "tif_path": file_path,
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude")
            })
    
    pred_input_df = pd.DataFrame(records)
    
    st.session_state["input_df"] = pred_input_df
    
    import zipfile
    
    ZIP_PATH = "esri_doubt_dataset.zip"
    
    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:
    
        for file in os.listdir(PATCH_DIR):
    
            z.write(
                os.path.join(
                    PATCH_DIR,
                    file
                ),
                arcname=file
            )
    
    with open(
        ZIP_PATH,
        "rb"
    ) as f:
    
        st.download_button(
            "Download Dataset",
            f,
            file_name="esri_doubt_dataset.zip",
            mime="application/zip"
        )
    
    st.success(
        f"{len(results)} patches generated."
    )
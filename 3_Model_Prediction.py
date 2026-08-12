# import torch
# import rasterio
# import numpy as np
# import pandas as pd
# import streamlit as st
# import folium

# from streamlit_folium import st_folium
# from collections import Counter 
# # ==========================================================
# # ESRI LABELS
# # ==========================================================
# task_type=st.session_state["task_type"]
# selected_models = st.session_state["selected_models"]

# if task_type=="binary":
#     ESRI_IDX_TO_LABEL = {
#         0: "others",
#         1: "vegetation"
#     }

#     SENT_IDX_TO_LABEL = {
#         0: "others",
#         1: "vegetation"
#     }
    
#     CLASS_COLORS = {
#         "others": "#808080",
#         "vegetation": "#00aa00"
#     }

# else:
#     ESRI_IDX_TO_LABEL = {
#         0: "factory",
#         1: "mine",
#         2: "solar",
#         3: "urban",
#         4: "vegetation"
#     }

#     SENT_IDX_TO_LABEL = {
#         0: "vegetation",
#         1: "urban",
#         2: "solar",
#         3: "factory",
#         4: "mine"
#     }
    
#     CLASS_COLORS = {
#         "factory": "#ff0000",
#         "mine": "#ff8c00",
#         "solar": "#ffd700",
#         "urban": "#0066ff",
#         "vegetation": "#00aa00"
#     }





# # ==========================================================
# # ESRI PREDICTION
# # ==========================================================

# def predict_esri_patch(tif_path):

#     with rasterio.open(tif_path) as src:

#         img = src.read([1, 2, 3])

#     x = torch.tensor(
#         img.astype(np.float32) / 255.0,
#         dtype=torch.float32
#     ).unsqueeze(0)

#     with torch.no_grad():

#         probs = torch.softmax(
#             st.session_state.esri_model(x),
#             dim=1
#         )[0]

#     conf, pred = torch.max(
#         probs,
#         dim=0
#     )

#     return {
#         "prediction":
#         ESRI_IDX_TO_LABEL[
#             pred.item()
#         ],

#         "confidence":
#         float(conf.item())
#     }


# # ==========================================================
# # SENTINEL PREDICTION
# # ==========================================================

# def predict_sentinel_patch(
#     mm_path,
#     od_path
# ):

#     with rasterio.open(mm_path) as src:
#         img1 = src.read([1, 2, 3, 4])

#     with rasterio.open(od_path) as src:
#         img2 = src.read([1, 2, 3, 4])

#     x1 = torch.tensor(
#         img1.astype(np.float32) / 255.0,
#         dtype=torch.float32
#     ).unsqueeze(0)

#     x2 = torch.tensor(
#         img2.astype(np.float32) / 255.0,
#         dtype=torch.float32
#     ).unsqueeze(0)

#     with torch.no_grad():

#         probs = torch.softmax(
#             st.session_state.sent_model(
#                 x1,
#                 x2
#             ),
#             dim=1
#         )[0]

#     conf, pred = torch.max(
#         probs,
#         dim=0
#     )

#     return {
#         "prediction":
#         SENT_IDX_TO_LABEL[
#             pred.item()
#         ],

#         "confidence":
#         float(conf.item())
#     }


# def majority_vote(results):

#     labels = [
#         r["prediction"]
#         for r in results
#         if r is not None
#     ]

#     if len(labels) == 0:
#         return "Missing"

#     return Counter(
#         labels
#     ).most_common(1)[0][0]


# def weighted_vote(results):

#     scores = {}

#     for r in results:

#         if r is None:
#             continue

#         label = r["prediction"]

#         scores[label] = (
#             scores.get(label, 0)
#             + r["confidence"]
#         )

#     if len(scores) == 0:
#         return "Missing"

#     return max(
#         scores,
#         key=scores.get
#     )


# # ==========================================================
# # RUN PREDICTION
# # ==========================================================

# if st.button(
#     "Run Predictions",
#     use_container_width=True
# ):

#     if "ESRI" in selected_models:

#         if "esri_model" not in st.session_state:
    
#             st.error(
#                 "ESRI model not loaded."
#             )
    
#             st.stop()
    
#         if "esri_patch_map" not in st.session_state:
    
#             st.error(
#                 "No ESRI patches found."
#             )
    
#             st.stop()
    
    
#     if "Sentinel" in selected_models:
    
#         if "sent_model" not in st.session_state:
    
#             st.error(
#                 "Sentinel model not loaded."
#             )
    
#             st.stop()
    
#         if "mm_patch_map" not in st.session_state:
    
#             st.error(
#                 "MM patches missing."
#             )
    
#             st.stop()
    
#         if "od_patch_map" not in st.session_state:
    
#             st.error(
#                 "OD patches missing."
#             )
    
#             st.stop()
    
#     if "ESRI" in selected_models:
    
#         esri_model = st.session_state.esri_model
#         esri_model.eval()
    
#     if "Sentinel" in selected_models:
    
#         sent_model = st.session_state.sent_model
#         sent_model.eval()
    
#     input_df = (
#         st.session_state["input_df"]
#         .copy()
#     )

#     esri_patch_map = {}
#     mm_patch_map = {}
#     od_patch_map = {}
    
#     if "ESRI" in selected_models:
    
#         esri_patch_map = st.session_state.get(
#             "esri_patch_map",
#             {}
#         )
    
#     if "Sentinel" in selected_models:
    
#         mm_patch_map = st.session_state.get(
#             "mm_patch_map",
#             {}
#         )
    
#         od_patch_map = st.session_state.get(
#             "od_patch_map",
#             {}
#         )

#     results = []

#     progress = st.progress(0)

#     total = len(input_df)

#     for count, (idx, row) in enumerate(
#         input_df.iterrows(),
#         start=1
#     ):

#         esri_pred = None
#         sent_pred = None
        
#         if "ESRI" in selected_models:
        
#             esri_path = esri_patch_map.get(idx)
        
#             if esri_path is not None:
        
#                 esri_pred = predict_esri_patch(
#                     esri_path
#                 )
        
#         if "Sentinel" in selected_models:
        
#             mm_path = mm_patch_map.get(idx)
        
#             od_path = od_patch_map.get(idx)
        
#             if (
#                 mm_path is not None
#                 and
#                 od_path is not None
#             ):
        
#                 sent_pred = predict_sentinel_patch(
#                     mm_path,
#                     od_path
#                 )

#         results_for_vote = []

#         if esri_pred is not None:
#             results_for_vote.append(
#                 esri_pred
#             )
        
#         if sent_pred is not None:
#             results_for_vote.append(
#                 sent_pred
#             )
        
#         majority_pred = majority_vote(
#             results_for_vote
#         )
        
#         weighted_pred = weighted_vote(
#             results_for_vote
#         )

#         if "latitude" in row and "longitude" in row:

#             latitude = row["latitude"]
#             longitude = row["longitude"]
        
#         elif "geometry" in row:
        
#             centroid = row.geometry.centroid
        
#             latitude = centroid.y
#             longitude = centroid.x
        
#         else:
        
#             continue

#         results.append({

#             "latitude":
#             latitude,
        
#             "longitude":
#             longitude,
        
#             "esri_prediction":
#             esri_pred["prediction"]
#             if esri_pred
#             else "NA",
        
#             "esri_confidence":
#             esri_pred["confidence"]
#             if esri_pred
#             else np.nan,
        
#             "sent_prediction":
#             sent_pred["prediction"]
#             if sent_pred
#             else "NA",
        
#             "sent_confidence":
#             sent_pred["confidence"]
#             if sent_pred
#             else np.nan,
        
#             "majority_prediction":
#             majority_pred,
        
#             "weighted_prediction":
#             weighted_pred
#         })

#         progress.progress(
#             count / total
#         )

#     pred_df = pd.DataFrame(
#         results
#     )

#     st.session_state[
#         "pred_df"
#     ] = pred_df

#     st.success(
#         f"{len(pred_df)} predictions completed."
#     )



# # ==========================================================
# # DISPLAY RESULTS
# # ==========================================================


    
# if "pred_df" in st.session_state:

#     pred_df = st.session_state[
#         "pred_df"
#     ]

#     st.subheader(
#         "Prediction Results"
#     )

#     st.dataframe(
#         pred_df,
#         use_container_width=True
#     )

#     # ------------------------------------------------------
#     # CLASS COUNTS
#     # ------------------------------------------------------

#     st.subheader(
#         "Class Distribution"
#     )

#     st.dataframe(
#         pred_df["weighted_prediction"]
#         .value_counts()
#         .reset_index()
#         .rename(
#             columns={
#                 "weighted_prediction": "Count"
#             }
#         )
#     )

#     # ------------------------------------------------------
#     # MAP
#     # ------------------------------------------------------

#     st.subheader(
#         "Prediction Map"
#     )

#     center_lat = (
#         pred_df["latitude"]
#         .mean()
#     )

#     center_lon = (
#         pred_df["longitude"]
#         .mean()
#     )

#     m = folium.Map(
#         location=[
#             center_lat,
#             center_lon
#         ],
#         zoom_start=11,
#         tiles=None
#     )

#     # ------------------------------------------------------
#     # ESRI WORLD IMAGERY
#     # ------------------------------------------------------

#     folium.TileLayer(
#         tiles=(
#             "https://server.arcgisonline.com/"
#             "ArcGIS/rest/services/"
#             "World_Imagery/MapServer/"
#             "tile/{z}/{y}/{x}"
#         ),
#         attr="Esri",
#         name="Esri Satellite"
#     ).add_to(m)

#     # ------------------------------------------------------
#     # PREDICTION POINTS
#     # ------------------------------------------------------

#     for _, row in pred_df.iterrows():

#         pred = row["weighted_prediction"]

#         color = CLASS_COLORS.get(
#             pred,
#             "black"
#         )

#         esri_conf = (
#             f"{row['esri_confidence']:.3f}"
#             if pd.notna(row["esri_confidence"])
#             else "NA"
#         )
        
#         sent_conf = (
#             f"{row['sent_confidence']:.3f}"
#             if pd.notna(row["sent_confidence"])
#             else "NA"
#         )

#         popup_html = f"""
#         <table style='width:300px'>
        
#         <tr>
#         <td><b>ESRI</b></td>
#         <td>{row['esri_prediction']}</td>
#         </tr>
        
#         <tr>
#         <td><b>ESRI Conf</b></td>
#         <td>{esri_conf}</td>
#         </tr>
        
#         <tr>
#         <td><b>Sentinel</b></td>
#         <td>{row['sent_prediction']}</td>
#         </tr>
        
#         <tr>
#         <td><b>Sentinel Conf</b></td>
#         <td>{sent_conf}</td>
#         </tr>
        
#         <tr>
#         <td><b>Majority</b></td>
#         <td>{row['majority_prediction']}</td>
#         </tr>
        
#         <tr>
#         <td><b>Weighted</b></td>
#         <td>{row['weighted_prediction']}</td>
#         </tr>
        
#         </table>
#         """

#         folium.CircleMarker(
#             location=[
#                 row["latitude"],
#                 row["longitude"]
#             ],
#             radius=8,
#             color=color,
#             fill=True,
#             fill_color=color,
#             fill_opacity=0.9,
#             weight=2,
#             tooltip=pred,
#             popup=folium.Popup(
#                 popup_html,
#                 max_width=300
#             )
#         ).add_to(m)

#     # ------------------------------------------------------
#     # LEGEND
#     # ------------------------------------------------------

#     if task_type=="binary":
#         legend_html = """
#         <div style="
#             position: fixed;
#             bottom: 40px;
#             left: 40px;
#             width: 200px;
#             background-color: white;
#             color:black;
#             border: 2px solid gray;
#             border-radius: 5px;
#             z-index: 9999;
#             padding: 10px;
#             font-size: 14px;
#         ">
    
#         <b>Classes</b>
    
#         <br><br>
    
#         <span style="color:#808080;">⬤</span>
#         Others
    
#         <br>
    
#         <span style="color:#00aa00;">⬤</span>
#         Vegetation
    
#         </div>
#         """
#     else:
#         legend_html = """
#         <div style="
#             position: fixed;
#             bottom: 40px;
#             left: 40px;
#             width: 200px;
#             background-color: white;
#             color:black;
#             border: 2px solid gray;
#             border-radius: 5px;
#             z-index: 9999;
#             padding: 10px;
#             font-size: 14px;
#         ">
    
#         <b>Classes</b>
    
#         <br><br>
    
#         <span style="color:#ff0000;">⬤</span>
#         Factory
    
#         <br>
    
#         <span style="color:#ff8c00;">⬤</span>
#         Mine
    
#         <br>
    
#         <span style="color:#ffd700;">⬤</span>
#         Solar
    
#         <br>
    
#         <span style="color:#0066ff;">⬤</span>
#         Urban
    
#         <br>
    
#         <span style="color:#00aa00;">⬤</span>
#         Vegetation
    
#         </div>
#         """

#     m.get_root().html.add_child(
#         folium.Element(
#             legend_html
#         )
#     )

#     folium.LayerControl().add_to(
#         m
#     )

#     st_folium(
#         m,
#         width=1400,
#         height=800,
#         returned_objects=[]
#     )

#     # ------------------------------------------------------
#     # DOWNLOAD CSV
#     # ------------------------------------------------------

#     csv = pred_df.to_csv(
#         index=False
#     )

#     st.download_button(
#         "Download Prediction CSV",
#         csv,
#         file_name="predictions.csv",
#         mime="text/csv",
#         use_container_width=True
#     )


import torch
import rasterio
import numpy as np
import pandas as pd
import streamlit as st
import folium

from streamlit_folium import st_folium
from collections import Counter 
# ==========================================================
# ESRI LABELS
# ==========================================================
task_type=st.session_state["task_type"]
selected_models = st.session_state["selected_models"]

if task_type=="binary":
    ESRI_IDX_TO_LABEL = {
        0: "others",
        1: "vegetation"
    }

    SENT_IDX_TO_LABEL = {
        0: "others",
        1: "vegetation"
    }
    
    CLASS_COLORS = {
        "others": "#808080",
        "vegetation": "#00aa00"
    }

else:
    ESRI_IDX_TO_LABEL = {
        0: "factory",
        1: "mine",
        2: "solar",
        3: "urban",
        4: "vegetation"
    }

    SENT_IDX_TO_LABEL = {
        0: "vegetation",
        1: "urban",
        2: "solar",
        3: "factory",
        4: "mine"
    }
    
    CLASS_COLORS = {
        "factory": "#ff0000",
        "mine": "#ff8c00",
        "solar": "#ffd700",
        "urban": "#0066ff",
        "vegetation": "#00aa00"
    }





# ==========================================================
# ESRI PREDICTION
# ==========================================================

def predict_esri_patch(tif_path):

    with rasterio.open(tif_path) as src:

        img = src.read([1, 2, 3])

    x = torch.tensor(
        img.astype(np.float32) / 255.0,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        probs = torch.softmax(
            st.session_state.esri_model(x),
            dim=1
        )[0]

    conf, pred = torch.max(
        probs,
        dim=0
    )

    return {
        "prediction":
        ESRI_IDX_TO_LABEL[
            pred.item()
        ],

        "confidence":
        float(conf.item())
    }


# ==========================================================
# SENTINEL PREDICTION
# ==========================================================

def predict_sentinel_patch(
    mm_path,
    od_path
):

    with rasterio.open(mm_path) as src:
        img1 = src.read([1, 2, 3, 4])

    with rasterio.open(od_path) as src:
        img2 = src.read([1, 2, 3, 4])

    x1 = torch.tensor(
        img1.astype(np.float32) / 255.0,
        dtype=torch.float32
    ).unsqueeze(0)

    x2 = torch.tensor(
        img2.astype(np.float32) / 255.0,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        probs = torch.softmax(
            st.session_state.sent_model(
                x1,
                x2
            ),
            dim=1
        )[0]

    conf, pred = torch.max(
        probs,
        dim=0
    )

    return {
        "prediction":
        SENT_IDX_TO_LABEL[
            pred.item()
        ],

        "confidence":
        float(conf.item())
    }

def predict_combined_patch(esri_path, mm_path, od_path):

    with rasterio.open(esri_path) as src:
        esri = src.read([1,2,3])

    with rasterio.open(mm_path) as src:
        mm = src.read([1,2,3,4])

    with rasterio.open(od_path) as src:
        od = src.read([1,2,3,4])

    esri = torch.tensor(
        esri.astype(np.float32)/255.
    ).unsqueeze(0)

    mm = torch.tensor(
        mm.astype(np.float32)/255.
    ).unsqueeze(0)

    od = torch.tensor(
        od.astype(np.float32)/255.
    ).unsqueeze(0)

    esri = torch.nn.functional.interpolate(
        esri,
        size=(224,224),
        mode="bilinear",
        align_corners=False
    )

    mm = torch.nn.functional.interpolate(
        mm,
        size=(224,224),
        mode="bilinear",
        align_corners=False
    )

    od = torch.nn.functional.interpolate(
        od,
        size=(224,224),
        mode="bilinear",
        align_corners=False
    )

    # esri=esri.to(DEVICE)
    # mm=mm.to(DEVICE)
    # od=od.to(DEVICE)

    with torch.no_grad():

        probs=torch.softmax(

            st.session_state.combined_model(
                esri,
                mm,
                od
            ),

            dim=1

        )[0]

    conf,pred=torch.max(probs,0)

    if task_type=="binary":

        label="others" if pred.item()==0 else "vegetation"

    else:

        label=ESRI_IDX_TO_LABEL[pred.item()]

    return {
        "prediction": label,
        "confidence": float(conf.item())
    }



def majority_vote(results):

    labels = [
        r["prediction"]
        for r in results
        if r is not None
    ]

    if len(labels) == 0:
        return "Missing"

    return Counter(
        labels
    ).most_common(1)[0][0]


def weighted_vote(results):

    scores = {}

    for r in results:

        if r is None:
            continue

        label = r["prediction"]

        scores[label] = (
            scores.get(label, 0)
            + r["confidence"]
        )

    if len(scores) == 0:
        return "Missing"

    return max(
        scores,
        key=scores.get
    )


# ==========================================================
# RUN PREDICTION
# ==========================================================

if st.button(
    "Run Predictions",
    use_container_width=True
):

    if ("ESRI" in selected_models) or ("Combined" in selected_models):

        if "esri_model" not in st.session_state:
    
            st.error(
                "ESRI model not loaded."
            )
    
            st.stop()
    
        if "esri_patch_map" not in st.session_state:
    
            st.error(
                "No ESRI patches found."
            )
    
            st.stop()
    
    
    if ("Sentinel" in selected_models) or ("Combined" in selected_models):
    
        if "sent_model" not in st.session_state:
    
            st.error(
                "Sentinel model not loaded."
            )
    
            st.stop()
    
        if "mm_patch_map" not in st.session_state:
    
            st.error(
                "MM patches missing."
            )
    
            st.stop()
    
        if "od_patch_map" not in st.session_state:
    
            st.error(
                "OD patches missing."
            )
    
            st.stop()

    if "Combined" in selected_models:

        if "combined_model" not in st.session_state:
    
            st.error("Combined model not loaded.")
            st.stop()
    
        if "esri_patch_map" not in st.session_state:
    
            st.error("No ESRI patches found.")
            st.stop()
    
        if "mm_patch_map" not in st.session_state:
    
            st.error("MM patches missing.")
            st.stop()
    
        if "od_patch_map" not in st.session_state:
    
            st.error("OD patches missing.")
            st.stop()
    
    if "ESRI" in selected_models:
    
        esri_model = st.session_state.esri_model
        esri_model.eval()
    
    if "Sentinel" in selected_models:
    
        sent_model = st.session_state.sent_model
        sent_model.eval()

    if "Combined" in selected_models:

        combined_model = st.session_state.combined_model
        combined_model.eval()
    
    input_df = (
        st.session_state["input_df"]
        .copy()
    )

    esri_patch_map = {}
    mm_patch_map = {}
    od_patch_map = {}
    
    if "ESRI" in selected_models:
    
        esri_patch_map = st.session_state.get(
            "esri_patch_map",
            {}
        )
    
    if "Sentinel" in selected_models:
    
        mm_patch_map = st.session_state.get(
            "mm_patch_map",
            {}
        )
    
        od_patch_map = st.session_state.get(
            "od_patch_map",
            {}
        )

    results = []

    progress = st.progress(0)

    total = len(input_df)

    for count, (idx, row) in enumerate(
        input_df.iterrows(),
        start=1
    ):

        esri_pred = None
        sent_pred = None
        combined_pred = None
        
        if "ESRI" in selected_models:
        
            esri_path = esri_patch_map.get(idx)
        
            if esri_path is not None:
        
                esri_pred = predict_esri_patch(
                    esri_path
                )
        
        if "Sentinel" in selected_models:
        
            mm_path = mm_patch_map.get(idx)
        
            od_path = od_patch_map.get(idx)
        
            if (
                mm_path is not None
                and
                od_path is not None
            ):
        
                sent_pred = predict_sentinel_patch(
                    mm_path,
                    od_path
                )

        if "Combined" in selected_models:

            esri_path = esri_patch_map.get(idx)
            mm_path = mm_patch_map.get(idx)
            od_path = od_patch_map.get(idx)
        
            if (
                esri_path is not None
                and mm_path is not None
                and od_path is not None
            ):
        
                combined_pred = predict_combined_patch(
                    esri_path,
                    mm_path,
                    od_path
                )
        
        results_for_vote = []

        if esri_pred is not None:
            results_for_vote.append(
                esri_pred
            )
        
        if sent_pred is not None:
            results_for_vote.append(
                sent_pred
            )

        if combined_pred is not None:
            results_for_vote.append(combined_pred)
        
        majority_pred = majority_vote(
            results_for_vote
        )
        
        weighted_pred = weighted_vote(
            results_for_vote
        )

        if "latitude" in row and "longitude" in row:

            latitude = row["latitude"]
            longitude = row["longitude"]
        
        elif "geometry" in row:
        
            centroid = row.geometry.centroid
        
            latitude = centroid.y
            longitude = centroid.x
        
        else:
        
            continue

        results.append({

            "latitude":
            latitude,
        
            "longitude":
            longitude,
        
            "esri_prediction":
            esri_pred["prediction"]
            if esri_pred
            else "NA",
        
            "esri_confidence":
            esri_pred["confidence"]
            if esri_pred
            else np.nan,
        
            "sent_prediction":
            sent_pred["prediction"]
            if sent_pred
            else "NA",
        
            "sent_confidence":
            sent_pred["confidence"]
            if sent_pred
            else np.nan,

            "combined_prediction":
            combined_pred["prediction"]
            if combined_pred
            else "NA",
            
            "combined_confidence":
            combined_pred["confidence"]
            if combined_pred
            else np.nan,
        
            "majority_prediction":
            majority_pred,
        
            "weighted_prediction":
            weighted_pred
        })

        progress.progress(
            count / total
        )

    pred_df = pd.DataFrame(
        results
    )

    st.session_state[
        "pred_df"
    ] = pred_df

    st.success(
        f"{len(pred_df)} predictions completed."
    )



# ==========================================================
# DISPLAY RESULTS
# ==========================================================


    
if "pred_df" in st.session_state:

    pred_df = st.session_state[
        "pred_df"
    ]

    st.subheader(
        "Prediction Results"
    )

    st.dataframe(
        pred_df,
        use_container_width=True
    )

    # ------------------------------------------------------
    # CLASS COUNTS
    # ------------------------------------------------------

    st.subheader(
        "Class Distribution"
    )

    st.dataframe(
        pred_df["weighted_prediction"]
        .value_counts()
        .reset_index()
        .rename(
            columns={
                "weighted_prediction": "Count"
            }
        )
    )

    # ------------------------------------------------------
    # MAP
    # ------------------------------------------------------

    st.subheader(
        "Prediction Map"
    )

    center_lat = (
        pred_df["latitude"]
        .mean()
    )

    center_lon = (
        pred_df["longitude"]
        .mean()
    )

    m = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=11,
        tiles=None
    )

    # ------------------------------------------------------
    # ESRI WORLD IMAGERY
    # ------------------------------------------------------

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Imagery/MapServer/"
            "tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Esri Satellite"
    ).add_to(m)

    # ------------------------------------------------------
    # PREDICTION POINTS
    # ------------------------------------------------------

    for _, row in pred_df.iterrows():

        pred = row["weighted_prediction"]

        color = CLASS_COLORS.get(
            pred,
            "black"
        )

        esri_conf = (
            f"{row['esri_confidence']:.3f}"
            if pd.notna(row["esri_confidence"])
            else "NA"
        )
        
        sent_conf = (
            f"{row['sent_confidence']:.3f}"
            if pd.notna(row["sent_confidence"])
            else "NA"
        )

        combined_conf = (
            f"{row['combined_confidence']:.3f}"
            if pd.notna(row["combined_confidence"])
            else "NA"
        )

        popup_html = f"""
        <table style='width:300px'>
        
        <tr>
        <td><b>ESRI</b></td>
        <td>{row['esri_prediction']}</td>
        </tr>
        
        <tr>
        <td><b>ESRI Conf</b></td>
        <td>{esri_conf}</td>
        </tr>
        
        <tr>
        <td><b>Sentinel</b></td>
        <td>{row['sent_prediction']}</td>
        </tr>
        
        <tr>
        <td><b>Sentinel Conf</b></td>
        <td>{sent_conf}</td>
        </tr>

        <tr>
        <td><b>Combined</b></td>
        <td>{row['combined_prediction']}</td>
        </tr>
        
        <tr>
        <td><b>Combined Conf</b></td>
        <td>{combined_conf}</td>
        </tr>
        
        <tr>
        <td><b>Majority</b></td>
        <td>{row['majority_prediction']}</td>
        </tr>
        
        <tr>
        <td><b>Weighted</b></td>
        <td>{row['weighted_prediction']}</td>
        </tr>
        
        </table>
        """

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=2,
            tooltip=pred,
            popup=folium.Popup(
                popup_html,
                max_width=300
            )
        ).add_to(m)

    # ------------------------------------------------------
    # LEGEND
    # ------------------------------------------------------

    if task_type=="binary":
        legend_html = """
        <div style="
            position: fixed;
            bottom: 40px;
            left: 40px;
            width: 200px;
            background-color: white;
            color:black;
            border: 2px solid gray;
            border-radius: 5px;
            z-index: 9999;
            padding: 10px;
            font-size: 14px;
        ">
    
        <b>Classes</b>
    
        <br><br>
    
        <span style="color:#808080;">⬤</span>
        Others
    
        <br>
    
        <span style="color:#00aa00;">⬤</span>
        Vegetation
    
        </div>
        """
    else:
        legend_html = """
        <div style="
            position: fixed;
            bottom: 40px;
            left: 40px;
            width: 200px;
            background-color: white;
            color:black;
            border: 2px solid gray;
            border-radius: 5px;
            z-index: 9999;
            padding: 10px;
            font-size: 14px;
        ">
    
        <b>Classes</b>
    
        <br><br>
    
        <span style="color:#ff0000;">⬤</span>
        Factory
    
        <br>
    
        <span style="color:#ff8c00;">⬤</span>
        Mine
    
        <br>
    
        <span style="color:#ffd700;">⬤</span>
        Solar
    
        <br>
    
        <span style="color:#0066ff;">⬤</span>
        Urban
    
        <br>
    
        <span style="color:#00aa00;">⬤</span>
        Vegetation
    
        </div>
        """

    m.get_root().html.add_child(
        folium.Element(
            legend_html
        )
    )

    folium.LayerControl().add_to(
        m
    )

    st_folium(
        m,
        width=1400,
        height=800,
        returned_objects=[]
    )

    # ------------------------------------------------------
    # DOWNLOAD CSV
    # ------------------------------------------------------

    csv = pred_df.to_csv(
        index=False
    )

    st.download_button(
        "Download Prediction CSV",
        csv,
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True
    )
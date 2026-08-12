import os
import glob
import re
from collections import Counter

import streamlit as st
import numpy as np
import pandas as pd
import rasterio
import kagglehub

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# CONFIG
# ==========================================================

MULTICLASS_NAMES = [
    "vegetation",
    "urban",
    "solar",
    "factory",
    "mine"
]

BINARY_NAMES = [
    "others",
    "vegetation"
]

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

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CSV_PATH = "feedback_labels.csv"


# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "i": 0,
    "samples": [],
    "current_sample": None,
    "models_loaded": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==========================================================
# TASK TYPE
# ==========================================================

task_type = st.radio(
    "Classification Task",
    [
        "multiclass",
        "binary"
    ]
)

if task_type == "multiclass":
    CLASS_NAMES = MULTICLASS_NAMES
else:
    CLASS_NAMES = BINARY_NAMES



# ==========================================================
# MODEL COMBINATION
# ==========================================================
selected_models = st.multiselect(
    "Models to use",
    [
        "ESRI",
        "Sentinel",
        "Combined"
    ],
    default=["Combined"]
)

# model_choice = st.selectbox(
#     "Models to Use",
#     [
#         "ESRI",
#         "MM",
#         "OD",
#         "ESRI + MM",
#         "ESRI + OD",
#         "MM + OD",
#         "ESRI + MM + OD"
#     ]
# )

st.session_state["task_type"] = task_type
st.session_state["selected_models"] = selected_models
# ==========================================================
# MODEL BUILDERS
# ==========================================================

def build_esri_model(num_classes):

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


class SiameseResNet18(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        backbone = models.resnet18(weights=None)

        backbone.conv1 = nn.Conv2d(
            4, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.encoder = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.fc = nn.Linear(512 * 3, num_classes)

    def forward(self, x1, x2):

        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        f1 = torch.flatten(f1, 1)
        f2 = torch.flatten(f2, 1)

        diff = torch.abs(f1 - f2)

        x = torch.cat([f1, f2, diff], dim=1)

        return self.fc(x)


class MultiModalResNet18(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        ###################################################
        # ESRI BACKBONE (3 CHANNEL)
        ###################################################

        esri_backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.esri_encoder = nn.Sequential(
            *list(esri_backbone.children())[:-1]
        )

        ###################################################
        # SENTINEL BACKBONE (4 CHANNEL)
        ###################################################

        sent_backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        old_conv = sent_backbone.conv1

        sent_backbone.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        ###################################################
        # Initialize 4th channel (NIR)
        ###################################################

        with torch.no_grad():

            sent_backbone.conv1.weight[:, :3] = old_conv.weight

            sent_backbone.conv1.weight[:, 3] = \
                old_conv.weight.mean(dim=1)

        self.sentinel_encoder = nn.Sequential(
            *list(sent_backbone.children())[:-1]
        )

        ###################################################
        # Feature Fusion
        ###################################################

        self.fusion = nn.Sequential(

            nn.Linear(512 * 4, 1024),

            nn.BatchNorm1d(1024),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(1024, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(512, num_classes)

        )

    def forward(self, esri, mm, od):

        ###############################################
        # ESRI
        ###############################################

        esri_feat = self.esri_encoder(esri)

        esri_feat = torch.flatten(
            esri_feat,
            1
        )

        ###############################################
        # March-May
        ###############################################

        mm_feat = self.sentinel_encoder(mm)

        mm_feat = torch.flatten(
            mm_feat,
            1
        )

        ###############################################
        # Oct-Dec
        ###############################################

        od_feat = self.sentinel_encoder(od)

        od_feat = torch.flatten(
            od_feat,
            1
        )

        ###############################################
        # Temporal Difference
        ###############################################
        temporal = torch.abs(
            mm_feat - od_feat
        )
        ###############################################
        # Fusion
        ###############################################
        features = torch.cat(
            [
                esri_feat,
                mm_feat,
                od_feat,
                temporal
            ],
            dim=1
        )
        return self.fusion(features)

# ==========================================================
# LOAD MODELS
# ==========================================================
@st.cache_resource
def load_models(
    task_type,
    use_esri,
    use_sent,
    use_combined,
    esri_url=None,
    sent_url=None,
    combined_url=None,
    version_tag="latest"
):

    num_classes = 2 if task_type == "binary" else 5

    def resolve_model(url):
        base_dir = kagglehub.model_download(url)

        all_pths = glob.glob(
            os.path.join(base_dir, "**/*.pth"),
            recursive=True
        )

        all_pths = sorted(all_pths)

        if version_tag == "latest":
            return all_pths[-1]

        for p in all_pths:
            if version_tag in p:
                return p

        return all_pths[-1]

    esri_model = None
    sent_model = None
    combined_model=None

    # ---------------- ESRI ----------------
    if use_esri:
        esri_path = resolve_model(esri_url)

        esri_model = build_esri_model(num_classes)
        esri_model.load_state_dict(torch.load(esri_path, map_location=DEVICE))
        esri_model.eval()

    # ---------------- SENTINEL (2-stream) ----------------
    if use_sent:
        sent_path = resolve_model(sent_url)

        sent_model = SiameseResNet18(num_classes=num_classes)
        sent_model.load_state_dict(torch.load(sent_path, map_location=DEVICE))
        sent_model.eval()

    if use_combined:

        combined_path = resolve_model(combined_url)
    
        combined_model = MultiModalResNet18(num_classes)
    
        combined_model.load_state_dict(
            torch.load(
                combined_path,
                map_location=DEVICE
            )
        )
    
        combined_model.eval()

    return esri_model, sent_model, combined_model
    
# ==========================================================
# MODEL URLS
# ==========================================================

st.subheader("Model Loading")

if task_type=="binary":
    esri_url = st.text_input(
        "ESRI model URL",
        value="chinmayeep2385/esri-binary-class/pyTorch/default"
    )
    
    sent_url = st.text_input(
        "Sentinel-2 model URL",
        value="chinmayeep2385/sent2-comb-binary/PyTorch/resnet-2stream-4band"
    )

    combined_url = st.text_input(
        "Combined model URL",
        value="chinmayeep2385/comb-binary/PyTorch/resnet-comb-4band"
    )
    
else:
    esri_url = st.text_input(
        "ESRI model URL",
        value="chinmayeep2385/esri-multi-class/pyTorch/class-weights"
    )
    
    sent_url = st.text_input(
        "Sentinel-2 model URL",
        value="chinmayeep2385/sent2-comb-multiclass/PyTorch/resnet-2stream-4band"
    )

    combined_url = st.text_input(
        "Combined model URL",
        value="chinmayeep2385/comb-multiclass/PyTorch/resnet-2stream-4band"
    )
 

version_tag = st.text_input(
    "Version tag",
    "latest"
)

if st.button("Load Models"):

    esri_model, sent_model, combined_model = load_models(
        task_type=task_type,
        use_esri=("ESRI" in selected_models),
        use_sent=("Sentinel" in selected_models),
        use_combined=("Combined" in selected_models),
        esri_url=esri_url,
        sent_url=sent_url,
        combined_url=combined_url,
        version_tag=version_tag
    )
    
    st.session_state.esri_model = esri_model
    st.session_state.sent_model = sent_model
    st.session_state.combined_model = combined_model

    st.session_state.models_loaded = True

    st.success("Models loaded successfully")

# ==========================================================
# DATASET URLS
# ==========================================================

st.subheader("Dataset Loading")

esri_ds = st.text_input(
    "ESRI dataset",
    value="chinmayeep2385/esri-doubt-dataset"
)

mm_ds = st.text_input(
    "MM dataset",
    value="chinmayeep2385/sent2-mm-doubt-dataset"
)

od_ds = st.text_input(
    "OD dataset",
    value="chinmayeep2385/sent2-od-doubt-dataset"
)


# ==========================================================
# GET TILE ID
# ==========================================================

def get_idx(path):

    m = re.search(
        r"(\d+)\.tif$",
        os.path.basename(path)
    )

    if m:
        return int(m.group(1))

    return -1


# ==========================================================
# LOAD DATASETS
# ==========================================================

if st.button("Load Datasets"):

    esri_dir = None
    mm_dir = None
    od_dir = None
    
    if ("ESRI" in selected_models) or ("Combined" in selected_models):
        esri_dir = kagglehub.dataset_download(esri_ds)
        st.session_state["esri_dir"]=esri_dir
    
    if ("Sentinel" in selected_models) or ("Combined" in selected_models):
        mm_dir = kagglehub.dataset_download(mm_ds)
        od_dir = kagglehub.dataset_download(od_ds)
        st.session_state["mm_dir"] = mm_dir
        st.session_state["od_dir"] = od_dir

    esri_files = []
    mm_files = []
    od_files = []
    
    if esri_dir is not None:
        esri_files = glob.glob(
            os.path.join(esri_dir, "**/*.tif"),
            recursive=True
        )
    
    if mm_dir is not None:
        mm_files = glob.glob(
            os.path.join(mm_dir, "**/*.tif"),
            recursive=True
        )
    
    if od_dir is not None:
        od_files = glob.glob(
            os.path.join(od_dir, "**/*.tif"),
            recursive=True
        )
        
    esri_map = {}
    mm_map = {}
    od_map = {}
    
    if ("ESRI" in selected_models) or ("Combined" in selected_models):
        esri_map = {
            get_idx(f): f
            for f in esri_files
        }
    
    if ("Sentinel" in selected_models) or ("Combined" in selected_models):
        mm_map = {
            get_idx(f): f
            for f in mm_files
        }
    
        od_map = {
            get_idx(f): f
            for f in od_files
        }

    id_sets = []

    if ("ESRI" in selected_models) or ("Combined" in selected_models):
        id_sets.append(set(esri_map))
    
    if ("Sentinel" in selected_models) or ("Combined" in selected_models):
        id_sets.append(set(mm_map))
        id_sets.append(set(od_map))
    
    common_ids = sorted(
        set.intersection(*id_sets)
    )

    samples = []
    
    for i in common_ids:
    
        sample = {}
    
        if ("ESRI" in selected_models) or ("Combined" in selected_models):
            sample["esri"] = esri_map[i]
    
        if ("Sentinel" in selected_models) or ("Combined" in selected_models):
            sample["mm"] = mm_map[i]
            sample["od"] = od_map[i]
    
        samples.append(sample)
    
    st.session_state.samples = samples

    st.session_state.i = 0

    st.success(
        f"{len(common_ids)} samples loaded"
    )


# ==========================================================
# IMAGE LOADER
# ==========================================================

def load_img(path):

    if path is None:
        return "Missing"

    if not os.path.exists(path):
        return "Missing"

    try:

        with rasterio.open(path) as src:

            img = src.read([1, 2, 3])

        img = np.transpose(
            img,
            (1, 2, 0)
        )

        img = np.clip(
            img,
            0,
            255
        ).astype(np.uint8)

        return img

    except Exception:

        return "Missing"


# ==========================================================
# BINARY CONVERSION
# ==========================================================

def binary_to_int(label):

    if label == "vegetation":
        return 1

    elif label == "others":
        return 0

    return label


# ==========================================================
# ESRI PREDICTION
# ==========================================================

def predict_esri(path):

    with rasterio.open(path) as src:

        img = src.read([1, 2, 3])

    x = torch.tensor(
        img.astype(np.float32) / 255.0,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        probs = torch.softmax(
            st.session_state.esri_model(x),
            dim=1
        )[0]

    conf, pred = torch.max(
        probs,
        dim=0
    )

    st.write(
        "Raw prediction index:",
        pred.item()
    )

    if task_type == "binary":

        pred_label = (
            "others"
            if pred.item() == 0
            else "vegetation"
        )
    
    else:
    
        pred_label = ESRI_IDX_TO_LABEL[
            pred.item()
        ]
    
    return {
        "pred": pred_label,
        "conf": float(conf.item())
    }


# ==========================================================
# SENTINEL PREDICTION
# ==========================================================

def predict_sentinel_pair(mm_path, od_path):

    with rasterio.open(mm_path) as src:
        img1 = src.read([1,2,3,4])

    with rasterio.open(od_path) as src:
        img2 = src.read([1,2,3,4])

    x1 = torch.tensor(img1.astype(np.float32)/255.0).unsqueeze(0).to(DEVICE)
    x2 = torch.tensor(img2.astype(np.float32)/255.0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(
            st.session_state.sent_model(x1, x2),
            dim=1
        )[0]

    conf, pred = torch.max(probs, dim=0)

    if task_type == "binary":
        pred_label = "others" if pred.item() == 0 else "vegetation"
    else:
        pred_label = SENT_IDX_TO_LABEL[pred.item()]

    return {
        "pred": pred_label,
        "conf": float(conf.item())
    }


def predict_combined(esri_path, mm_path, od_path):

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

    esri=esri.to(DEVICE)
    mm=mm.to(DEVICE)
    od=od.to(DEVICE)

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
        "pred":label,
        "conf":float(conf)
    }
# ==========================================================
# MAJORITY VOTE
# ==========================================================

def majority_vote(results):

    labels = []

    for r in results:

        if r is not None:

            labels.append(
                r["pred"]
            )

    if len(labels) == 0:
        return "Missing"

    return Counter(
        labels
    ).most_common(1)[0][0]

# ==========================================================
# WEIGHTED VOTE
# ==========================================================

def weighted_vote(results):

    valid_results = [
        r
        for r in results
        if r is not None
    ]

    if len(valid_results) == 0:
        return "Missing"


    # ---------------- binary ----------------
    if task_type == "binary":

        scores = {
            c: 0.0
            for c in BINARY_NAMES
        }

    # ---------------- multiclass ----------------
    else:

        scores = {
            c: 0.0
            for c in MULTICLASS_NAMES
        }


    for r in valid_results:

        scores[
            r["pred"]
        ] += r["conf"]


    return max(
        scores,
        key=scores.get
    )
# ==========================================================
# GET ACTIVE MODEL RESULTS
# ==========================================================

def get_predictions(sample):

    results = []

    esri_res = None
    sent_res=None
    combined_res=None

    if "ESRI" in selected_models:

        esri_res = predict_esri(
            sample["esri"]
        )

        results.append(esri_res)

    if "Sentinel" in selected_models:
    
        sent_res = predict_sentinel_pair(
            sample["mm"],
            sample["od"]
        )
    
        results.append(sent_res)

    if "Combined" in selected_models:

        combined_res = predict_combined(
            sample["esri"],
            sample["mm"],
            sample["od"]
        )
        results.append(combined_res)
    
    return (
        esri_res,
        sent_res,
        combined_res,
        results
    )


# ==========================================================
# REFRESH
# ==========================================================

def refresh():

    if (
        not st.session_state.models_loaded
        or len(st.session_state.samples) == 0
    ):

        return (
            [],
            "",
            "",
            [],
            None,
            None,
            None
        )

    i = st.session_state.i

    if i >= len(st.session_state.samples):

        return (
            [],
            "DONE",
            "",
            [],
            None,
            None,
            None
        )

    sample = st.session_state.samples[i]

    st.session_state.current_sample = sample

    esri_res, sent_res, combined_res, results = get_predictions(
        sample
    )

    majority = majority_vote(results)

    weighted = weighted_vote(results)

    images = []

    texts = []


    if "esri" in sample:

        images.append(
            (
                "ESRI",
                load_img(
                    sample.get("esri")
                )
            )
        )


    if "mm" in sample:

        images.append(
            (
                "Sentinel MM",
                load_img(
                    sample.get("mm")
                )
            )
        )

    if "od" in sample:

        images.append(
            (
                "Sentinel OD",
                load_img(
                    sample.get("od")
                )
            )
        )


    if esri_res is not None:
        st.info(
            f"ESRI : {esri_res['pred']} ({esri_res['conf']:.3f})"
        )
    
    if sent_res is not None:
        st.info(
            f"Sentinel : {sent_res['pred']} ({sent_res['conf']:.3f})"
        )
    
    if combined_res is not None:
        st.success(
            f"Combined : {combined_res['pred']} ({combined_res['conf']:.3f})"
        )

    vote_txt = (
        f"Majority : {majority}\n\n"
        f"Weighted : {weighted}"
    )


    filename = os.path.basename(
        next(
            iter(sample.values())
        )
    )


    return (
        images,
        filename,
        vote_txt,
        texts,
        esri_res,
        sent_res,
        combined_res
    )

# ==========================================================
# SAVE CSV
# ==========================================================

def save_label(
    sample,
    label
):

    esri_res, sent_res, combined_res, results = get_predictions(
        sample
    )

    row = {

        "filename":
        os.path.basename(
            next(iter(sample.values()))
        ),

        "task":
        task_type,

        "models_used":
        selected_models,

        "esri_pred":
        esri_res["pred"] if esri_res else "NA",

        "esri_conf":
        esri_res["conf"]
        if esri_res
        else "NA",

        # MM
        "sent_pred":
        sent_res["pred"] if sent_res else "NA",

        "sent_conf":
        sent_res["conf"]
        if sent_res
        else "NA",

        "combined_pred":
        combined_res["pred"] if combined_res else "NA",
        
        "combined_conf":
        combined_res["conf"] if combined_res else "NA",

        "majority_pred":
        majority_vote(results),

        "weighted_pred":
        weighted_vote(results),

        "correct_label":
        label
    }

    df = pd.DataFrame(
        [row]
    )

    if os.path.exists(
        CSV_PATH
    ):

        df.to_csv(
            CSV_PATH,
            mode="a",
            header=False,
            index=False
        )

    else:

        df.to_csv(
            CSV_PATH,
            index=False
        )


# ==========================================================
# LABEL ACTION
# ==========================================================

def label_action(
    label
):

    sample = st.session_state.current_sample

    if sample is None:
        return

    save_label(
        sample,
        label
    )

    st.session_state.i += 1

    st.rerun()

# ==========================================================
# DISPLAY CURRENT SAMPLE
# ==========================================================

if (
    st.session_state.models_loaded
    and len(st.session_state.samples) > 0
    and st.session_state.i < len(st.session_state.samples)
):

    (
        images,
        filename,
        vote_txt,
        pred_texts,
        esri_res,
        sent_res,
        combined_res
    ) = refresh()

    st.subheader(filename)

    n_imgs = len(images)

    if n_imgs == 1:

        col = st.columns(1)

        with col[0]:

            title, img = images[0]

            if img is not None:
                st.image(
                    img,
                    caption=title
                )

            # st.write(
            #     pred_texts[0]
            # )

    elif n_imgs == 2:

        cols = st.columns(2)

        for i in range(2):

            with cols[i]:

                title, img = images[i]

                if img is not None:

                    st.image(
                        img,
                        caption=title
                    )

                # st.write(
                #     pred_texts[i]
                # )

    elif n_imgs == 3:

        cols = st.columns(3)

        for i in range(3):

            with cols[i]:

                title, img = images[i]

                if isinstance(img, str):

                    st.warning(
                        f"{title} image missing"
                    )
                
                else:
                
                    st.image(
                        img,
                        caption=title
                    )
                
                # st.write(
                #     pred_texts[i]
                # )

    st.info(
        vote_txt
    )

    st.divider()


# ==========================================================
# LABEL BUTTONS
# ==========================================================

if task_type == "binary":

    labels = [
        "others",
        "vegetation",
        "skip"
    ]

else:

    labels = [
        "factory",
        "mine",
        "solar",
        "urban",
        "vegetation",
        "skip"
    ]


cols = st.columns(
    len(labels)
)

for col, label in zip(
    cols,
    labels
):

    with col:

        if st.button(
            label,
            use_container_width=True
        ):

            label_action(
                label
            )


# ==========================================================
# PROGRESS BAR
# ==========================================================

total = max(
    len(
        st.session_state.samples
    ),
    1
)

progress = (
    st.session_state.i
    / total
)

st.progress(
    progress
)

st.write(
    f"{st.session_state.i}/{len(st.session_state.samples)} completed"
)


# ==========================================================
# DOWNLOAD CSV
# ==========================================================

if os.path.exists(
    CSV_PATH
):

    with open(
        CSV_PATH,
        "rb"
    ) as f:

        st.download_button(
            "Download Labels CSV",
            data=f,
            file_name="feedback_labels.csv",
            mime="text/csv"
        )


# ==========================================================
# EVALUATION METRICS
# ==========================================================
st.divider()

if st.button("View Evaluation Metrics"):

    if not os.path.exists(CSV_PATH):

        st.warning("No feedback_labels.csv found.")

    else:

        df = pd.read_csv(CSV_PATH)

        # remove skipped samples
        df = df[df["correct_label"] != "skip"]

        if len(df) == 0:

            st.warning("No labelled samples available.")

        else:

            prediction_columns = [
                "esri_pred",
                "sent_pred",
                "combined_pred",
                "majority_pred",
                "weighted_pred"
            ]
            
            for pred_column in prediction_columns:
            
                if pred_column not in df.columns:
                    continue
            
                temp = df[df[pred_column] != "NA"]
            
                y_true = temp["correct_label"]
                y_pred = temp[pred_column]
            
                acc = accuracy_score(
                    y_true,
                    y_pred
                )
            
                st.subheader(pred_column)
            
                st.metric(
                    "Accuracy",
                    f"{acc*100:.2f}%"
                )
            
                report_df = pd.DataFrame(
                    classification_report(
                        y_true,
                        y_pred,
                        output_dict=True
                    )
                ).transpose()
            
                st.dataframe(
                    report_df,
                    use_container_width=True
                )

                temp = df[
                    (df[pred_column] != "NA")
                    &
                    (df["correct_label"] != "skip")
                ]
                
                y_true = temp["correct_label"].astype(str)
                y_pred = temp[pred_column].astype(str)
                
                labels = sorted(
                    set(y_true).union(set(y_pred))
                )
                
                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=labels
                )
                
                cm_df = pd.DataFrame(
                    cm,
                    index=labels,
                    columns=labels
                )
    
                st.subheader("Confusion Matrix")
    
                st.dataframe(
                    cm_df,
                    use_container_width=True
                )

# ==========================================================
# DONE
# ==========================================================

if (
    len(
        st.session_state.samples
    ) > 0
    and
    st.session_state.i >= len(
        st.session_state.samples
    )
):

    st.success(
        "Labeling complete."
    )
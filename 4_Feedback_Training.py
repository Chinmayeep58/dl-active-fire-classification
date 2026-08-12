# import os
# import glob
# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import rasterio
# import streamlit as st
# import folium
# import re
# from streamlit_folium import st_folium
# from torchvision import models
# from torch.utils.data import Dataset, DataLoader
# import kagglehub
# from datetime import datetime
# import json

# DEVICE = torch.device(
#     "cuda"
#     if torch.cuda.is_available()
#     else "cpu"
# )

# if st.session_state["task_type"]=="binary":
#     feedback_csv_url = st.text_input(
#         "Feedback CSV Dataset",
#         value="chinmayeep2385/feedback-dataset-binary"
#     )
# else:
#     feedback_csv_url = st.text_input(
#         "Feedback CSV Dataset",
#         value="chinmayeep2385/esri-feedback-dataset"
#     )

# if st.button("Load Feedback CSV"):

#     csv_dir = kagglehub.dataset_download(
#         feedback_csv_url
#     )

#     csv_path = glob.glob(
#         os.path.join(
#             csv_dir,
#             "**/*.csv"
#         ),
#         recursive=True
#     )[0]

#     st.session_state["feedback_csv"] = csv_path

#     st.success(
#         "Feedback CSV loaded."
#     )
# selected_models = st.session_state["selected_models"]

# def build_esri_model(num_classes):

#     model = models.resnet18(
#         weights=None
#     )

#     model.fc = nn.Linear(
#         model.fc.in_features,
#         num_classes
#     )

#     return model


# class SiameseResNet18(nn.Module):

#     def __init__(self, num_classes):

#         super().__init__()

#         backbone = models.resnet18(weights=None)

#         backbone.conv1 = nn.Conv2d(
#             4,
#             64,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False
#         )

#         self.encoder = nn.Sequential(
#             *list(backbone.children())[:-1]
#         )

#         self.fc = nn.Linear(
#             512 * 3,
#             num_classes
#         )

#     def forward(self, x1, x2):

#         f1 = self.encoder(x1)
#         f2 = self.encoder(x2)

#         f1 = torch.flatten(f1, 1)
#         f2 = torch.flatten(f2, 1)

#         diff = torch.abs(
#             f1 - f2
#         )

#         x = torch.cat(
#             [
#                 f1,
#                 f2,
#                 diff
#             ],
#             dim=1
#         )

#         return self.fc(x)

# if st.session_state["task_type"] == "binary":

#     ESRI_LABEL_TO_IDX = {
#         "others": 0,
#         "vegetation": 1
#     }

#     SENT_LABEL_TO_IDX = {
#         "others": 0,
#         "vegetation": 1
#     }

# else:

#     ESRI_LABEL_TO_IDX = {
#         "factory":0,
#         "mine":1,
#         "solar":2,
#         "urban":3,
#         "vegetation":4
#     }

#     SENT_LABEL_TO_IDX = {
#         "vegetation":0,
#         "urban":1,
#         "solar":2,
#         "factory":3,
#         "mine":4
#     }

# class EsriFeedbackDataset(Dataset):

#     def __init__(
#         self,
#         df,
#         folder,
#         label_to_idx,
#         is_esri=True
#     ):

#         self.df = df[
#             df["correct_label"] != "skip"
#         ]

#         self.folder = folder
#         self.is_esri = is_esri
#         self.label_to_idx = label_to_idx



#     def __len__(self):

#         return len(self.df)


#     def __getitem__(self, idx):

#         row = self.df.iloc[idx]
    
#         filename = row["filename"]
    
#         matches = glob.glob(
#             os.path.join(
#                 self.folder,
#                 "**",
#                 filename
#             ),
#             recursive=True
#         )
    
#         if len(matches) == 0:
#             raise FileNotFoundError(filename)
    
#         path = matches[0]
    
#         with rasterio.open(path) as src:
#             img = src.read([1,2,3])
    
#         img = img.astype(np.float32) / 255.0
    
#         x = torch.tensor(
#             img,
#             dtype=torch.float32
#         )
    
#         label = str(
#             row["correct_label"]
#         ).strip().lower()
    
#         y = torch.tensor(
#             self.label_to_idx[label],
#             dtype=torch.long
#         )
    
#         return x, y


# class SentinelFeedbackDataset(Dataset):

#     def __init__(
#         self,
#         df,
#         label_to_idx
#     ):

#         self.df = df[
#             df["correct_label"] != "skip"
#         ]

#         self.label_to_idx = label_to_idx

#     def __len__(self):

#         return len(self.df)

#     def __getitem__(self, idx):

#         row = self.df.iloc[idx]
    
#         filename = row["filename"]
    
#         tile_id = re.search(
#             r"(\d+)\.tif$",
#             filename
#         ).group(1)
    
#         mm_matches = glob.glob(
#             os.path.join(
#                 st.session_state["mm_dir"],
#                 f"**/*{tile_id}.tif"
#             ),
#             recursive=True
#         )
    
#         od_matches = glob.glob(
#             os.path.join(
#                 st.session_state["od_dir"],
#                 f"**/*{tile_id}.tif"
#             ),
#             recursive=True
#         )
    
#         if len(mm_matches) == 0:
#             raise FileNotFoundError(
#                 f"MM tile {tile_id}"
#             )
    
#         if len(od_matches) == 0:
#             raise FileNotFoundError(
#                 f"OD tile {tile_id}"
#             )
    
#         mm_path = mm_matches[0]
#         od_path = od_matches[0]
    
#         with rasterio.open(mm_path) as src:
#             mm = src.read([1,2,3,4])
    
#         with rasterio.open(od_path) as src:
#             od = src.read([1,2,3,4])
    
#         mm = torch.tensor(
#             mm.astype(np.float32)/255.0,
#             dtype=torch.float32
#         )
    
#         od = torch.tensor(
#             od.astype(np.float32)/255.0,
#             dtype=torch.float32
#         )
    
#         label = str(
#             row["correct_label"]
#         ).strip().lower()
    
#         y = torch.tensor(
#             self.label_to_idx[label],
#             dtype=torch.long
#         )
    
#         return mm, od, y


# def prepare_model(model):

#     for p in model.parameters():

#         p.requires_grad = False


#     for p in model.layer4.parameters():

#         p.requires_grad = True


#     for p in model.fc.parameters():

#         p.requires_grad = True


#     return model

# def save_versioned_model(
#     model,
#     model_name,
#     task_type,
#     feedback_count
# ):

#     timestamp = datetime.now().strftime(
#         "%Y%m%d_%H%M%S"
#     )

#     version_name = (
#         f"{model_name}_"
#         f"{task_type}_"
#         f"fb{feedback_count}_"
#         f"{timestamp}.pth"
#     )

#     save_dir = "saved_models"

#     os.makedirs(
#         save_dir,
#         exist_ok=True
#     )

#     model_path = os.path.join(
#         save_dir,
#         version_name
#     )

#     torch.save(
#         model.state_dict(),
#         model_path
#     )

#     metadata = {
#         "model": model_name,
#         "task_type": task_type,
#         "feedback_samples": feedback_count,
#         "created": timestamp,
#         "filename": version_name
#     }

#     with open(
#         model_path.replace(
#             ".pth",
#             ".json"
#         ),
#         "w"
#     ) as f:

#         json.dump(
#             metadata,
#             f,
#             indent=4
#         )

#     return model_path

# def train_model(
#     model,
#     loader,
#     epochs=5,
#     lr=1e-4
# ):

#     model = prepare_model(
#         model
#     )

#     criterion = nn.CrossEntropyLoss()

#     optimizer = torch.optim.Adam(

#         filter(
#             lambda p:
#             p.requires_grad,
#             model.parameters()
#         ),

#         lr=lr
#     )

#     model.train()
#     loss_box = st.empty()

#     for epoch in range(epochs):

#         total_loss = 0

#         for x, y in loader:

#             x = x.to(DEVICE)
#             y = y.to(DEVICE)

#             optimizer.zero_grad()

#             out = model(x)

#             loss = criterion(
#                 out,
#                 y
#             )

#             loss.backward()

#             optimizer.step()

#             total_loss += loss.item()

#         epoch_loss = total_loss / len(loader)

#         print(
#             f"Epoch {epoch+1}/{epochs}  Loss = {epoch_loss:.4f}", flush=True
#         )
    
#         loss_box.write(
#             f"Epoch {epoch+1}/{epochs}  Loss = {epoch_loss:.4f}"
#         )

#     model.eval()

#     return model


# def train_siamese_model(
#     model,
#     loader,
#     epochs=5,
#     lr=1e-4
# ):

#     criterion = nn.CrossEntropyLoss()

#     optimizer = torch.optim.Adam(

#         filter(
#             lambda p: p.requires_grad,
#             model.parameters()
#         ),

#         lr=lr
#     )

#     model.train()

#     loss_box = st.empty()

#     for epoch in range(epochs):

#         total_loss = 0

#         for mm, od, y in loader:

#             mm = mm.to(DEVICE)
#             od = od.to(DEVICE)
#             y = y.to(DEVICE)

#             optimizer.zero_grad()

#             out = model(
#                 mm,
#                 od
#             )

#             loss = criterion(
#                 out,
#                 y
#             )

#             loss.backward()

#             optimizer.step()

#             total_loss += loss.item()

#         epoch_loss = (
#             total_loss
#             /
#             len(loader)
#         )

#         loss_box.write(
#             f"Epoch {epoch+1}/{epochs}  Loss = {epoch_loss:.4f}"
#         )

#     model.eval()

#     return model



# if (
#     "feedback_csv" in st.session_state
#     and
#     st.button(
#         "Start Feedback Training"
#     )
# ):

#     df = pd.read_csv(
#         st.session_state[
#             "feedback_csv"
#         ]
#     )

#     df = df[
#         df["correct_label"] != "skip"
#     ]

    

#     if (
#         "ESRI" in selected_models
#         and
#         "esri_model" in st.session_state
#         and
#         "esri_dir" in st.session_state
#     ):

#         st.subheader(
#             "Training ESRI"
#         )

#         ds = EsriFeedbackDataset(

#             df,

#             st.session_state[
#                 "esri_dir"
#             ],
#             ESRI_LABEL_TO_IDX,
#             True
#         )

#         loader = DataLoader(
#             ds,
#             batch_size=16,
#             shuffle=True
#         )

#         st.session_state.esri_model = train_model(
#             st.session_state.esri_model,
#             loader
#         )
        
#         esri_model_path = save_versioned_model(
#             st.session_state.esri_model,
#             "esri",
#             st.session_state["task_type"],
#             len(df)
#         )
        
#         st.session_state[
#             "latest_esri_model_path"
#         ] = esri_model_path

#     if (
#         "Sentinel" in selected_models
#         and
#         "sent_model" in st.session_state
#         and
#         "mm_dir" in st.session_state
#         and
#         "od_dir" in st.session_state
#     ):
    
#         st.subheader(
#             "Training Sentinel"
#         )
    
#         ds = SentinelFeedbackDataset(
    
#             df,
    
#             SENT_LABEL_TO_IDX
#         )
    
#         loader = DataLoader(
    
#             ds,
    
#             batch_size=16,
    
#             shuffle=True
#         )
    
#         st.session_state.sent_model = train_siamese_model(
#             st.session_state.sent_model,
#             loader
#         )
        
#         sent_model_path = save_versioned_model(
#             st.session_state.sent_model,
#             "sentinel",
#             st.session_state["task_type"],
#             len(df)
#         )
        
#         st.session_state[
#             "latest_sent_model_path"
#         ] = sent_model_path
 
#     st.success(
#         "Feedback training complete."
#     )

# st.divider()

# if "latest_esri_model_path" in st.session_state:

#     with open(
#         st.session_state[
#             "latest_esri_model_path"
#         ],
#         "rb"
#     ) as f:

#         st.download_button(
#             "Download Updated ESRI Model",
#             f,
#             file_name=os.path.basename(
#                 st.session_state[
#                     "latest_esri_model_path"
#                 ]
#             )
#         )

# if "latest_sent_model_path" in st.session_state:

#     with open(
#         st.session_state[
#             "latest_sent_model_path"
#         ],
#         "rb"
#     ) as f:

#         st.download_button(
#             "Download Updated Sentinel Model",
#             f,
#             file_name=os.path.basename(
#                 st.session_state[
#                     "latest_sent_model_path"
#                 ]
#             )
#         )






import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import rasterio
import streamlit as st
import folium
import re
from streamlit_folium import st_folium
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import kagglehub
from datetime import datetime
import json

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if st.session_state["task_type"]=="binary":
    feedback_csv_url = st.text_input(
        "Feedback CSV Dataset",
        value="chinmayeep2385/feedback-dataset-binary"
    )
else:
    feedback_csv_url = st.text_input(
        "Feedback CSV Dataset",
        value="chinmayeep2385/esri-feedback-dataset"
    )

if st.button("Load Feedback CSV"):

    csv_dir = kagglehub.dataset_download(
        feedback_csv_url
    )

    csv_path = glob.glob(
        os.path.join(
            csv_dir,
            "**/*.csv"
        ),
        recursive=True
    )[0]

    st.session_state["feedback_csv"] = csv_path

    st.success(
        "Feedback CSV loaded."
    )
selected_models = st.session_state["selected_models"]

def build_esri_model(num_classes):

    model = models.resnet18(
        weights=None
    )

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
            4,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.encoder = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.fc = nn.Linear(
            512 * 3,
            num_classes
        )

    def forward(self, x1, x2):

        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        f1 = torch.flatten(f1, 1)
        f2 = torch.flatten(f2, 1)

        diff = torch.abs(
            f1 - f2
        )

        x = torch.cat(
            [
                f1,
                f2,
                diff
            ],
            dim=1
        )

        return self.fc(x)

if st.session_state["task_type"] == "binary":

    ESRI_LABEL_TO_IDX = {
        "others": 0,
        "vegetation": 1
    }

    SENT_LABEL_TO_IDX = {
        "others": 0,
        "vegetation": 1
    }

else:

    ESRI_LABEL_TO_IDX = {
        "factory":0,
        "mine":1,
        "solar":2,
        "urban":3,
        "vegetation":4
    }

    SENT_LABEL_TO_IDX = {
        "vegetation":0,
        "urban":1,
        "solar":2,
        "factory":3,
        "mine":4
    }

class EsriFeedbackDataset(Dataset):

    def __init__(
        self,
        df,
        folder,
        label_to_idx,
        is_esri=True
    ):

        self.df = df[
            df["correct_label"] != "skip"
        ]

        self.folder = folder
        self.is_esri = is_esri
        self.label_to_idx = label_to_idx



    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]
    
        filename = row["filename"]
    
        matches = glob.glob(
            os.path.join(
                self.folder,
                "**",
                filename
            ),
            recursive=True
        )
    
        if len(matches) == 0:
            raise FileNotFoundError(filename)
    
        path = matches[0]
    
        with rasterio.open(path) as src:
            img = src.read([1,2,3])
    
        img = img.astype(np.float32) / 255.0
    
        x = torch.tensor(
            img,
            dtype=torch.float32
        )
    
        label = str(
            row["correct_label"]
        ).strip().lower()
    
        y = torch.tensor(
            self.label_to_idx[label],
            dtype=torch.long
        )
    
        return x, y


class SentinelFeedbackDataset(Dataset):

    def __init__(
        self,
        df,
        label_to_idx
    ):

        self.df = df[
            df["correct_label"] != "skip"
        ]

        self.label_to_idx = label_to_idx

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
    
        filename = row["filename"]
    
        tile_id = re.search(
            r"(\d+)\.tif$",
            filename
        ).group(1)
    
        mm_matches = glob.glob(
            os.path.join(
                st.session_state["mm_dir"],
                f"**/*{tile_id}.tif"
            ),
            recursive=True
        )
    
        od_matches = glob.glob(
            os.path.join(
                st.session_state["od_dir"],
                f"**/*{tile_id}.tif"
            ),
            recursive=True
        )
    
        if len(mm_matches) == 0:
            raise FileNotFoundError(
                f"MM tile {tile_id}"
            )
    
        if len(od_matches) == 0:
            raise FileNotFoundError(
                f"OD tile {tile_id}"
            )
    
        mm_path = mm_matches[0]
        od_path = od_matches[0]
    
        with rasterio.open(mm_path) as src:
            mm = src.read([1,2,3,4])
    
        with rasterio.open(od_path) as src:
            od = src.read([1,2,3,4])
    
        mm = torch.tensor(
            mm.astype(np.float32)/255.0,
            dtype=torch.float32
        )
    
        od = torch.tensor(
            od.astype(np.float32)/255.0,
            dtype=torch.float32
        )
    
        label = str(
            row["correct_label"]
        ).strip().lower()
    
        y = torch.tensor(
            self.label_to_idx[label],
            dtype=torch.long
        )
    
        return mm, od, y



class CombinedFeedbackDataset(Dataset):

    def __init__(self, df, label_to_idx):

        self.df = df[df["correct_label"] != "skip"].reset_index(drop=True)

        self.label_to_idx = label_to_idx

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        filename = row["filename"]

        tile_id = re.search(
            r"(\d+)\.tif$",
            filename
        ).group(1)

        #################################################
        # ESRI
        #################################################

        esri_matches = glob.glob(
            os.path.join(
                st.session_state["esri_dir"],
                "**",
                filename
            ),
            recursive=True
        )

        #################################################
        # MM
        #################################################

        mm_matches = glob.glob(
            os.path.join(
                st.session_state["mm_dir"],
                f"**/*{tile_id}.tif"
            ),
            recursive=True
        )

        #################################################
        # OD
        #################################################

        od_matches = glob.glob(
            os.path.join(
                st.session_state["od_dir"],
                f"**/*{tile_id}.tif"
            ),
            recursive=True
        )

        if len(esri_matches) == 0:
            raise FileNotFoundError(filename)

        if len(mm_matches) == 0:
            raise FileNotFoundError(f"MM {tile_id}")

        if len(od_matches) == 0:
            raise FileNotFoundError(f"OD {tile_id}")

        esri_path = esri_matches[0]
        mm_path = mm_matches[0]
        od_path = od_matches[0]

        #################################################

        with rasterio.open(esri_path) as src:
            esri = src.read([1,2,3]).astype(np.float32)

        with rasterio.open(mm_path) as src:
            mm = src.read([1,2,3,4]).astype(np.float32)

        with rasterio.open(od_path) as src:
            od = src.read([1,2,3,4]).astype(np.float32)

        #################################################

        esri /= 255.0
        mm /= 255.0
        od /= 255.0

        esri = torch.tensor(esri)

        esri = torch.nn.functional.interpolate(
            esri.unsqueeze(0),
            size=(224,224),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        mm = torch.tensor(mm)

        mm = torch.nn.functional.interpolate(
            mm.unsqueeze(0),
            size=(224,224),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        od = torch.tensor(od)

        od = torch.nn.functional.interpolate(
            od.unsqueeze(0),
            size=(224,224),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        label = str(
            row["correct_label"]
        ).strip().lower()

        y = torch.tensor(
            self.label_to_idx[label],
            dtype=torch.long
        )

        return esri, mm, od, y


def prepare_model(model):

    for p in model.parameters():

        p.requires_grad = False


    for p in model.layer4.parameters():

        p.requires_grad = True


    for p in model.fc.parameters():

        p.requires_grad = True


    return model


def prepare_combined_model(model):

    for p in model.parameters():
        p.requires_grad = False

    ###################################################
    # ESRI encoder
    ###################################################

    for p in model.esri_encoder[-1].parameters():
        p.requires_grad = True

    ###################################################
    # Sentinel encoder
    ###################################################

    for p in model.sentinel_encoder[-1].parameters():
        p.requires_grad = True

    ###################################################
    # Fusion head
    ###################################################

    for p in model.fusion.parameters():
        p.requires_grad = True

    return model


def save_versioned_model(
    model,
    model_name,
    task_type,
    feedback_count
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    version_name = (
        f"{model_name}_"
        f"{task_type}_"
        f"fb{feedback_count}_"
        f"{timestamp}.pth"
    )

    save_dir = "saved_models"

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    model_path = os.path.join(
        save_dir,
        version_name
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    metadata = {
        "model": model_name,
        "task_type": task_type,
        "feedback_samples": feedback_count,
        "created": timestamp,
        "filename": version_name
    }

    with open(
        model_path.replace(
            ".pth",
            ".json"
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    return model_path

def train_model(
    model,
    loader,
    epochs=5,
    lr=1e-4
):

    model = prepare_model(
        model
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(

        filter(
            lambda p:
            p.requires_grad,
            model.parameters()
        ),

        lr=lr
    )

    model.train()
    loss_box = st.empty()

    for epoch in range(epochs):

        total_loss = 0

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(
                out,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch+1}/{epochs}  Loss = {epoch_loss:.4f}", flush=True
        )
    
        loss_box.write(
            f"Epoch {epoch+1}/{epochs}  Loss = {epoch_loss:.4f}"
        )

    model.eval()

    return model


def train_siamese_model(
    model,
    loader,
    epochs=5,
    lr=1e-4
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=lr
    )

    model.train()

    loss_box = st.empty()

    for epoch in range(epochs):

        total_loss = 0

        for mm, od, y in loader:

            mm = mm.to(DEVICE)
            od = od.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            out = model(
                mm,
                od
            )

            loss = criterion(
                out,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        epoch_loss = (
            total_loss
            /
            len(loader)
        )

        loss_box.write(
            f"Epoch {epoch+1}/{epochs}  Loss = {epoch_loss:.4f}"
        )

    model.eval()

    return model



def train_combined_model(
    model,
    loader,
    epochs=5,
    lr=1e-4
):

    model = prepare_combined_model(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=lr
    )

    model.train()

    loss_box = st.empty()

    for epoch in range(epochs):

        total_loss = 0

        for esri, mm, od, y in loader:

            esri = esri.to(DEVICE)
            mm = mm.to(DEVICE)
            od = od.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            out = model(
                esri,
                mm,
                od
            )

            loss = criterion(
                out,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(loader)

        loss_box.write(
            f"Epoch {epoch+1}/{epochs}  Loss = {epoch_loss:.4f}"
        )

    model.eval()

    return model


if (
    "feedback_csv" in st.session_state
    and
    st.button(
        "Start Feedback Training"
    )
):

    df = pd.read_csv(
        st.session_state[
            "feedback_csv"
        ]
    )

    df = df[
        df["correct_label"] != "skip"
    ]

    

    if (
        "ESRI" in selected_models
        and
        "esri_model" in st.session_state
        and
        "esri_dir" in st.session_state
    ):

        st.subheader(
            "Training ESRI"
        )

        ds = EsriFeedbackDataset(

            df,

            st.session_state[
                "esri_dir"
            ],
            ESRI_LABEL_TO_IDX,
            True
        )

        loader = DataLoader(
            ds,
            batch_size=16,
            shuffle=True
        )

        st.session_state.esri_model = train_model(
            st.session_state.esri_model,
            loader
        )
        
        esri_model_path = save_versioned_model(
            st.session_state.esri_model,
            "esri",
            st.session_state["task_type"],
            len(df)
        )
        
        st.session_state[
            "latest_esri_model_path"
        ] = esri_model_path

    if (
        "Sentinel" in selected_models
        and
        "sent_model" in st.session_state
        and
        "mm_dir" in st.session_state
        and
        "od_dir" in st.session_state
    ):
    
        st.subheader(
            "Training Sentinel"
        )
    
        ds = SentinelFeedbackDataset(
    
            df,
    
            SENT_LABEL_TO_IDX
        )
    
        loader = DataLoader(
    
            ds,
    
            batch_size=16,
    
            shuffle=True
        )
    
        st.session_state.sent_model = train_siamese_model(
            st.session_state.sent_model,
            loader
        )
        
        sent_model_path = save_versioned_model(
            st.session_state.sent_model,
            "sentinel",
            st.session_state["task_type"],
            len(df)
        )
        
        st.session_state[
            "latest_sent_model_path"
        ] = sent_model_path

    ############################################################
    # COMBINED FEEDBACK TRAINING
    ############################################################
    
    if (
        "Combined" in selected_models
        and
        "combined_model" in st.session_state
        and
        "esri_dir" in st.session_state
        and
        "mm_dir" in st.session_state
        and
        "od_dir" in st.session_state
    ):
    
        st.subheader("Training Combined Model")
    
        ds = CombinedFeedbackDataset(
    
            df,
    
            ESRI_LABEL_TO_IDX
        )
    
        loader = DataLoader(
    
            ds,
    
            batch_size=16,
    
            shuffle=True
    
        )
    
        st.session_state.combined_model = train_combined_model(
    
            st.session_state.combined_model,
    
            loader
    
        )
    
        combined_model_path = save_versioned_model(
    
            st.session_state.combined_model,
    
            "combined",
    
            st.session_state["task_type"],
    
            len(df)
    
        )
    
        st.session_state[
            "latest_combined_model_path"
        ] = combined_model_path
    
    st.success(
        "Feedback training complete."
    )
st.divider()

if "latest_esri_model_path" in st.session_state:

    with open(
        st.session_state[
            "latest_esri_model_path"
        ],
        "rb"
    ) as f:

        st.download_button(
            "Download Updated ESRI Model",
            f,
            file_name=os.path.basename(
                st.session_state[
                    "latest_esri_model_path"
                ]
            )
        )

if "latest_sent_model_path" in st.session_state:

    with open(
        st.session_state[
            "latest_sent_model_path"
        ],
        "rb"
    ) as f:

        st.download_button(
            "Download Updated Sentinel Model",
            f,
            file_name=os.path.basename(
                st.session_state[
                    "latest_sent_model_path"
                ]
            )
        )

if "latest_combined_model_path" in st.session_state:
    
    with open(
        st.session_state["latest_combined_model_path"],
        "rb"
    ) as f:
        st.download_button(
            "Download Updated Combined Model",
            f,
            file_name=os.path.basename(
                st.session_state["latest_combined_model_path"]
            )
        )
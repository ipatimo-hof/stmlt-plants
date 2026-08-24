import streamlit as st
from PIL import Image, ImageOps, ExifTags
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
import json
import os

# ---------- Page config ----------
st.set_page_config(layout="wide", page_title="Plant Recognizer")
st.image("header.png", width="stretch")
st.write("# Pflanzen auf dem Gründach erkennen V.1.0 (PlantNet)")

# ---------- Device ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Transforms (ImageNet) ----------
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------- Load model + labels (cached) ----------
@st.cache_resource
def load_model_and_labels():
    # 1. Model weights
    weights_path = hf_hub_download(
        repo_id="cpoisson/plantnet300k-mobilenetv3-small",
        filename="mobilenetv3_small_v2.pth",
    )

    model = models.mobilenet_v3_small(weights=None, num_classes=1081)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # 2. Species names (from the companion repo that ships the mapping)
    labels_path = hf_hub_download(
        repo_id="cpoisson/plantnet300k-resnet18",
        filename="plantnet300K_species_id_2_name.json",
    )
    with open(labels_path, "r", encoding="utf-8") as f:
        species_id_to_name = json.load(f)

    # Sort by numeric species_id → list of 1081 names in class-index order
    sorted_ids = sorted(species_id_to_name.keys(), key=lambda x: int(x))
    class_names = [species_id_to_name[sid] for sid in sorted_ids]

    return model, class_names


model, class_names = load_model_and_labels()


# ---------- Helpers ----------
def correct_image_orientation(image: Image.Image) -> Image.Image:
    try:
        exif = image._getexif()
        if not exif:
            return image
        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orientation_key and orientation_key in exif:
            orientation = exif[orientation_key]
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image


def predict_plant(image: Image.Image, top_k: int = 3):
    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k=top_k)

    results = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_idxs.cpu().tolist()):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        results.append((name, float(prob)))
    return results


def display_results(image: Image.Image, names_and_probabilities):
    # bad plants list (lowercase scientific / common names)
    bad_plants = []
    if os.path.exists("plants.txt"):
        with open("plants.txt", "r", encoding="utf-8") as f:
            content = f.read()
            bad_plants = [p.strip().lower() for p in content.replace("\n", ",").split(",") if p.strip()]

    for name, prob in names_and_probabilities:
        name_lower = name.lower()
        # Wikipedia likes the pure species name without author citation
        wiki_name = name.split(" L.")[0].split(" (")[0].strip().replace(" ", "_")

        output = (
            f"Die Pflanze auf dem Bild ist möglicherweise **{name}** "
            f"mit einer Wahrscheinlichkeit von **{prob*100:.1f}%**."
        )

        if any(bad in name_lower for bad in bad_plants):
            output += " ⚠️ Diese Pflanze ist eine Gefahr für das Gründach!"
            st.markdown(f"<span style='color:#c0392b'>{output}</span>", unsafe_allow_html=True)
        else:
            output += " Diese Art wurde nicht in der Liste der nicht empfohlenen Gründach-Arten gefunden."
            st.markdown(output)

        st.markdown(f"[Mehr über {name}](https://de.wikipedia.org/wiki/{wiki_name})")

    st.image(image, width=400)


# ---------- Input UI ----------
if "captured_image" not in st.session_state:
    st.session_state["captured_image"] = None

camera_image = st.camera_input("Foto aufnehmen")
if camera_image is not None:
    st.session_state["captured_image"] = camera_image

uploaded_file = st.file_uploader("Oder Bild hochladen", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.session_state["captured_image"] = uploaded_file

if st.session_state["captured_image"] is not None:
    image = Image.open(st.session_state["captured_image"])
    image = correct_image_orientation(image)
    results = predict_plant(image, top_k=3)
    display_results(image, results)

# ---------- Disclaimer ----------
st.write("---")
st.write("**Ungeeignete Anwendungsfälle:**")
st.write("1. Diese App eignet sich nicht zur Bestimmung, ob eine Pflanze essbar, giftig oder toxisch ist.")
st.write("2. Diese App eignet sich nicht zur Bestimmung, ob die Pflanze medizinische Anwendungen hat.")
st.write("3. Diese App eignet sich nicht zur Bestimmung des Standorts des Benutzers basierend auf den sichtbaren Pflanzen.")
st.caption("Modell: PlantNet-300K · MobileNetV3-Small (1081 Arten)")
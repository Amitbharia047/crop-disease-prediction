"""
dataset.py — AgriVision Keras Dataset Loader & Pipeline
─────────────────────────────────────────────────────────────────
Provides dataset loading, data augmentation, class mappings,
and preprocessing pipelines for the PlantVillage crop disease benchmark dataset.

Benchmark Dataset:
  - PlantVillage Dataset (54,305 images across 38 crop disease classes and 14 species)
  - Classes include: Apple, Blueberry, Cherry, Corn, Grape, Orange,
    Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
─────────────────────────────────────────────────────────────────
"""

import os
from typing import Dict, List, Tuple, Any

# ──────────────────────────────────────────────────────────────
# 38 PLANTVILLAGE CLASSES (Standard Benchmark)
# ──────────────────────────────────────────────────────────────
PLANT_VILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Rich metadata for agronomic diagnostics, treatments, and severity
DISEASE_METADATA: Dict[str, Dict[str, Any]] = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "display_name": "Apple Scab (Venturia inaequalis)",
        "severity": "Moderate",
        "symptoms": "Olive-green to black velvety spots on leaves, premature defoliation, and scabby fruit lesions.",
        "organic_treatment": "Spray sulfur or liquid copper fungicide at bud break. Remove and compost fallen infected leaves.",
        "chemical_treatment": "Apply myclobutanil or captan fungicide according to manufacturer instructions.",
        "prevention": "Prune canopy annually to maximize air movement and accelerate leaf drying."
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "display_name": "Apple Black Rot (Botryosphaeria obtusa)",
        "severity": "Severe",
        "symptoms": "Frogeye leaf spots with purple margins, necrotic branch cankers, and dark shriveled mummified fruit.",
        "organic_treatment": "Prune out diseased twigs and remove all mummified apples from the tree and orchard floor.",
        "chemical_treatment": "Apply thiophanate-methyl or captan from pink bud stage through harvest.",
        "prevention": "Avoid tree stress and prevent mechanical wounds to tree bark."
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "display_name": "Cedar Apple Rust (Gymnosporangium juniperi-virginianae)",
        "severity": "Moderate",
        "symptoms": "Bright yellow-orange circular leaf lesions with small dark spots on the upper leaf surface.",
        "organic_treatment": "Apply copper or sulfur fungicides early in the spring as cedar galls begin swelling.",
        "chemical_treatment": "Apply myclobutanil or propiconazole fungicide at pink bud stage.",
        "prevention": "Remove eastern red cedar trees within 1–2 miles of orchards where feasible, or plant rust-resistant apple cultivars."
    },
    "Apple___healthy": {
        "plant": "Apple",
        "display_name": "Apple - Healthy Leaf",
        "severity": "None",
        "symptoms": "Vibrant uniform green foliage with no fungal lesions, necrosis, or chlorosis.",
        "organic_treatment": "No treatment required. Maintain balanced nitrogen-potassium soil nutrition.",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain regular pruning schedule and balanced soil moisture."
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "display_name": "Blueberry - Healthy Foliage",
        "severity": "None",
        "symptoms": "Healthy dark green leaves, acidic soil tolerance, no fungal spotting.",
        "organic_treatment": "No treatment required. Maintain soil pH between 4.5 and 5.2 with organic sulfur.",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain pine-bark mulch and drip irrigation."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "display_name": "Cherry Powdery Mildew (Podosphaera clandestina)",
        "severity": "Moderate",
        "symptoms": "White powdery fungal mycelium covering young leaves and shoot tips, causing leaf curling.",
        "organic_treatment": "Apply horticultural neem oil, potassium bicarbonate, or sulfur sprays at first sign.",
        "chemical_treatment": "Apply myclobutanil or quinoxyfen according to agricultural label rates.",
        "prevention": "Avoid excessive nitrogen fertilization which produces susceptible tender shoot growth."
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "display_name": "Cherry - Healthy Foliage",
        "severity": "None",
        "symptoms": "Glossy green leaves with uniform coloration and sturdy petiole attachment.",
        "organic_treatment": "No treatment required.",
        "chemical_treatment": "None needed.",
        "prevention": "Apply annual dormant oil spray to prevent overwintering mites."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "display_name": "Corn Gray Leaf Spot (Cercospora zeae-maydis)",
        "severity": "Severe",
        "symptoms": "Rectangular tan-to-gray leaf lesions bounded strictly by parallel leaf veins.",
        "organic_treatment": "Deeply incorporate crop residues after harvest to accelerate residue decomposition.",
        "chemical_treatment": "Apply strobilurin or triazole fungicides at tassel emergence (VT stage) if risk is high.",
        "prevention": "Plant resistant hybrid maize seed varieties and rotate away from continuous corn."
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "display_name": "Corn Common Rust (Puccinia sorghi)",
        "severity": "Moderate",
        "symptoms": "Raised brownish-red to cinnamon pustules (uredinia) scattered across upper and lower leaf surfaces.",
        "organic_treatment": "Ensure adequate potassium levels to enhance natural plant defense mechanisms.",
        "chemical_treatment": "Apply azoxystrobin or propiconazole foliar fungicide if pustules appear before flowering.",
        "prevention": "Select rust-resistant corn hybrids (Rp-gene resistance)."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "display_name": "Northern Corn Leaf Blight (Exserohilum turcicum)",
        "severity": "Severe",
        "symptoms": "Long, elliptical cigar-shaped grayish-green lesions that turn tan and can coalesce to kill large leaf areas.",
        "organic_treatment": "Till soil to bury residue where fungal conidia overwinter.",
        "chemical_treatment": "Apply pyraclostrobin or fluxapyroxad fungicide early in reproductive stages.",
        "prevention": "Use 2-year crop rotation with non-host crops such as soybeans or alfalfa."
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "display_name": "Corn - Healthy Foliage",
        "severity": "None",
        "symptoms": "Broad, deep green leaves with strong venation and vigorous vegetative vigor.",
        "organic_treatment": "No treatment needed. Keep side-dress nitrogen schedule.",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain balanced N-P-K soil fertility."
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "display_name": "Grape Black Rot (Guignardia bidwellii)",
        "severity": "Severe",
        "symptoms": "Small brown circular leaf lesions with dark margins and tiny black pycnidia; berries turn into shriveled black mummies.",
        "organic_treatment": "Prune out diseased canes and cultivate soil beneath trellises to bury mummies.",
        "chemical_treatment": "Apply mancozeb, captan, or tebuconazole starting when new shoots are 1-3 inches long.",
        "prevention": "Ensure aggressive canopy training (VSP trellis) for rapid sun exposure and drying."
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "display_name": "Grape Esca / Black Measles (Phaeomoniella chlamydospora)",
        "severity": "Severe",
        "symptoms": "Tiger-stripe interveinal leaf chlorosis and necrosis; dark brown spots on berries; internal trunk vascular staining.",
        "organic_treatment": "Paint pruning wounds immediately with biocontrol trichoderma paste.",
        "chemical_treatment": "Apply wound protectants containing thiophanate-methyl on fresh vine pruning cuts.",
        "prevention": "Prune late in the dormant season during dry conditions to reduce spore infection."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "display_name": "Grape Leaf Blight (Pseudocercospora vitis)",
        "severity": "Moderate",
        "symptoms": "Irregular dark reddish-brown leaf spots that can cause extensive late-season defoliation.",
        "organic_treatment": "Apply copper oxychloride sprays post-bloom.",
        "chemical_treatment": "Apply foliar strobilurin or chlorothalonil fungicides.",
        "prevention": "Prompt harvest and post-harvest canopy aeration."
    },
    "Grape___healthy": {
        "plant": "Grape",
        "display_name": "Grape - Healthy Vine",
        "severity": "None",
        "symptoms": "Lush palmate green foliage with clean venation and thriving fruit clusters.",
        "organic_treatment": "No treatment required.",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain vine canopy thinning and adequate potassium fertilization."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "display_name": "Citrus Greening / Huanglongbing (Candidatus Liberibacter)",
        "severity": "Severe",
        "symptoms": "Asymmetric blotchy mottled leaf chlorosis, yellow shoots, small lopsided bitter green fruits.",
        "organic_treatment": "Control Asian citrus psyllid vectors using predatory parasitoid wasps (Tamarixia radiata) and insecticidal soaps.",
        "chemical_treatment": "Apply targeted systemic insecticides (imidacloprid) for psyllid vector eradication.",
        "prevention": "Use certified disease-free nursery stock in screened propagation facilities."
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "display_name": "Peach Bacterial Spot (Xanthomonas arboricola)",
        "severity": "Severe",
        "symptoms": "Small angular water-soaked leaf spots turning purple-black; ragged 'shot-hole' leaf appearance.",
        "organic_treatment": "Apply copper bactericide sprays at leaf fall and early bud stages.",
        "chemical_treatment": "Apply oxytetracycline antibiotic sprays during bloom and petal fall.",
        "prevention": "Plant resistant cultivars and avoid overhead sprinkler irrigation."
    },
    "Peach___healthy": {
        "plant": "Peach",
        "display_name": "Peach - Healthy Leaf",
        "severity": "None",
        "symptoms": "Long lanceolate leaves with glossy green surface and no bacterial shot-holes.",
        "organic_treatment": "No treatment needed.",
        "chemical_treatment": "None needed.",
        "prevention": "Apply dormant copper spray for leaf curl prevention."
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Pepper",
        "display_name": "Bell Pepper Bacterial Spot (Xanthomonas euvesicatoria)",
        "severity": "Severe",
        "symptoms": "Water-soaked lesions on foliage turning brownish-black with chlorotic yellow halos; blistered fruit spots.",
        "organic_treatment": "Apply fixed copper tank-mixed with mancozeb or Bacillus subtilis biological bactericide.",
        "chemical_treatment": "Apply actigard (acibenzolar-S-methyl) for plant systemic acquired resistance.",
        "prevention": "Use certified pathogen-free seed; practice 3-year non-solanaceous crop rotation."
    },
    "Pepper,_bell___healthy": {
        "plant": "Pepper",
        "display_name": "Bell Pepper - Healthy Plant",
        "severity": "None",
        "symptoms": "Smooth vibrant green foliage, strong stems, and healthy blossom set.",
        "organic_treatment": "No treatment needed.",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain consistent drip irrigation and calcium nutrition."
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "display_name": "Potato Early Blight (Alternaria solani)",
        "severity": "Moderate",
        "symptoms": "Concentric ring 'target-board' brown-to-black spots on older leaves, surrounded by yellow chlorotic margins.",
        "organic_treatment": "Apply copper hydroxide or bio-fungicide Serenade (Bacillus amyloliquefaciens).",
        "chemical_treatment": "Apply chlorothalonil, azoxystrobin, or difenoconazole at first detection.",
        "prevention": "Avoid leaf wetness by using drip irrigation; maintain balanced plant nitrogen."
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "display_name": "Potato Late Blight (Phytophthora infestans)",
        "severity": "Severe",
        "symptoms": "Rapidly expanding water-soaked dark lesions on leaves and stems; white fuzzy sporulation on underside during high humidity.",
        "organic_treatment": "Destroy severely infected plants immediately. Apply copper sulfate preventative sprays.",
        "chemical_treatment": "Apply cymoxanil, mandipropamid, or fluopicolide fungicides immediately.",
        "prevention": "Plant certified disease-free seed tubers; monitor local blight forecast warnings."
    },
    "Potato___healthy": {
        "plant": "Potato",
        "display_name": "Potato - Healthy Canopy",
        "severity": "None",
        "symptoms": "Robust dark green compound foliage with sturdy stems and no blighted lesions.",
        "organic_treatment": "No treatment required.",
        "chemical_treatment": "None needed.",
        "prevention": "Hill soil properly around tubers to protect from spore wash and sunlight."
    },
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "display_name": "Raspberry - Healthy Cane",
        "severity": "None",
        "symptoms": "Pinnately compound green leaves with serrated margins and healthy canes.",
        "organic_treatment": "No treatment required.",
        "chemical_treatment": "None needed.",
        "prevention": "Prune out spent floricanes after summer fruiting."
    },
    "Soybean___healthy": {
        "plant": "Soybean",
        "display_name": "Soybean - Healthy Foliage",
        "severity": "None",
        "symptoms": "Trifoliate leaves with uniform chlorophyll distribution and active nodulation.",
        "organic_treatment": "No treatment required.",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain proper row spacing and scout regularly for aphids."
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "display_name": "Squash Powdery Mildew (Podosphaera xanthii)",
        "severity": "Moderate",
        "symptoms": "Powdery white talcum-powder-like fungal patches on leaf surfaces and petioles, leading to yellowing and crisping.",
        "organic_treatment": "Spray potassium bicarbonate (3 tbsp/gal water) or diluted milk spray (40% milk / 60% water).",
        "chemical_treatment": "Apply myclobutanil or triflumizole fungicide early in disease manifestation.",
        "prevention": "Plant mildew-resistant squash cultivars and provide ample plant spacing."
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "display_name": "Strawberry Leaf Scorch (Diplocarpon earlianum)",
        "severity": "Moderate",
        "symptoms": "Irregular purple-to-dark-brown spots without white centers; leaves curl upward and appear scorched/burned.",
        "organic_treatment": "Remove scorched foliage after renovation. Apply copper fungicide in early spring.",
        "chemical_treatment": "Apply captan or thiophanate-methyl fungicide during spring emergence.",
        "prevention": "Avoid overhead sprinkler irrigation and remove old infected leaves during post-harvest renovation."
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "display_name": "Strawberry - Healthy Leaf",
        "severity": "None",
        "symptoms": "Deep emerald green trifoliate foliage with clean serrated margins and active runner growth.",
        "organic_treatment": "No treatment needed.",
        "chemical_treatment": "None needed.",
        "prevention": "Mulch with clean straw to keep leaves and berries off bare soil."
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "display_name": "Tomato Bacterial Spot (Xanthomonas perforans)",
        "severity": "Severe",
        "symptoms": "Small, dark, water-soaked circular leaf lesions often surrounded by yellow halos, causing defoliation.",
        "organic_treatment": "Spray fixed copper mixed with Bacillus subtilis or biological bacteriophages.",
        "chemical_treatment": "Apply copper hydroxide + mancozeb combination for bactericidal synergy.",
        "prevention": "Never work in wet fields; sanitize all stakes, pruning shears, and greenhouse trays."
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "display_name": "Tomato Early Blight (Alternaria solani)",
        "severity": "Moderate",
        "symptoms": "Target-shaped concentric rings within brown spots on lower leaves, progressing upward with yellow chlorosis.",
        "organic_treatment": "Prune lower 12 inches of foliage. Apply copper octanoate or copper sulfate sprays.",
        "chemical_treatment": "Apply chlorothalonil, azoxystrobin, or pyraclostrobin on a 7–14 day schedule.",
        "prevention": "Mulch around plant bases to prevent soil splashing onto lower foliage."
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "display_name": "Tomato Late Blight (Phytophthora infestans)",
        "severity": "Severe",
        "symptoms": "Large, greasy water-soaked dark patches that turn brown and papery; white fungal growth on underside in damp weather.",
        "organic_treatment": "Immediately bag and remove infected vines. Apply preventative copper fungicides.",
        "chemical_treatment": "Apply cymoxanil, dimethomorph, or propamocarb systemic fungicides.",
        "prevention": "Destroy volunteer potatoes and tomatoes; choose late-blight resistant varieties (e.g. Mountain Magic)."
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "display_name": "Tomato Leaf Mold (Passalora fulva)",
        "severity": "Moderate",
        "symptoms": "Pale greenish-yellow spots on upper leaf surface with olive-brown velvety fungal mold underneath.",
        "organic_treatment": "Dramatically improve greenhouse ventilation. Apply biofungicide or copper soap.",
        "chemical_treatment": "Apply chlorothalonil or thiophanate-methyl if relative humidity exceeds 85%.",
        "prevention": "Keep greenhouse humidity below 85% with exhaust fans and horizontal airflow."
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "display_name": "Tomato Septoria Leaf Spot (Septoria lycopersici)",
        "severity": "Moderate",
        "symptoms": "Numerous small circular spots (1-3mm) with dark brown borders and ash-gray centers dotted with tiny black pycnidia.",
        "organic_treatment": "Remove spotted leaves from bottom upward. Spray copper soap every 7 days.",
        "chemical_treatment": "Apply chlorothalonil or mancozeb at the first sign of lower leaf spotting.",
        "prevention": "Practice 3-year crop rotation; stake plants and avoid overhead watering."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "display_name": "Tomato Two-Spotted Spider Mite (Tetranychus urticae)",
        "severity": "Moderate",
        "symptoms": "Fine yellow stippling/speckling on leaves, dense silky webbing between leaf joints and flower clusters.",
        "organic_treatment": "Release predatory mites (Phytoseiulus persimilis). Apply insecticidal soap or neem oil.",
        "chemical_treatment": "Apply bifenazate or abamectin miticide targeted to leaf undersides.",
        "prevention": "Avoid dusty conditions and over-fertilizing with nitrogen, which stimulates mite reproduction."
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "display_name": "Tomato Target Spot (Corynespora cassiicola)",
        "severity": "Moderate",
        "symptoms": "Brown circular lesions with faint concentric rings and light brown centers, frequently causing blossom drop.",
        "organic_treatment": "Improve air circulation with aggressive pruning. Apply copper fungicides.",
        "chemical_treatment": "Apply fluopyram or boscalid fungicide early in the infection cycle.",
        "prevention": "Disinfect cages and stakes between growing seasons; maintain dry foliage."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "display_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "severity": "Severe",
        "symptoms": "Severe stunting, cupped erect leaflets, yellowing margins, and total failure of fruit set.",
        "organic_treatment": "Manage silverleaf whitefly vectors using yellow sticky traps, floating row covers, and insecticidal soaps.",
        "chemical_treatment": "Apply systemic neonicotinoid or diamide insecticides for whitefly control.",
        "prevention": "Plant TYLCV-resistant hybrids; install 50-mesh insect screens in nurseries."
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "display_name": "Tomato Mosaic Virus (ToMV)",
        "severity": "Severe",
        "symptoms": "Mottled light and dark green mosaic patterns on foliage, distorted 'shoestring' leaves, and stunted plant vigor.",
        "organic_treatment": "Remove and incinerate infected plants. Wash hands with whole milk or trisodium phosphate (TSP) after handling.",
        "chemical_treatment": "No chemical virucides exist. Focus on sanitation and vector exclusion.",
        "prevention": "Smokers must wash hands before touching plants (tobacco harbors the virus); choose resistant seed."
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "display_name": "Tomato - Healthy Plant",
        "severity": "None",
        "symptoms": "Deep green compound leaves with healthy trichomes (leaf hairs), robust stems, and no viral or fungal spotting.",
        "organic_treatment": "No treatment required. Maintain balanced 5-10-10 or organic compost tea feedings.",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain regular watering and calcium levels to prevent blossom end rot."
    }
}


def get_dataset_classes() -> List[str]:
    """Returns the list of 38 standard PlantVillage benchmark classes."""
    return PLANT_VILLAGE_CLASSES.copy()


def get_disease_metadata(class_name: str) -> Dict[str, Any]:
    """
    Returns rich agronomic metadata for a given PlantVillage class,
    including plant species, symptoms, severity, and treatments.
    """
    if class_name in DISEASE_METADATA:
        return DISEASE_METADATA[class_name]
    
    # Clean fallback for any arbitrary class format
    plant = class_name.split("___")[0].replace("_", " ") if "___" in class_name else "Plant"
    disease = class_name.split("___")[-1].replace("_", " ") if "___" in class_name else class_name
    return {
        "plant": plant,
        "display_name": f"{plant} - {disease}",
        "severity": "Moderate" if "healthy" not in class_name.lower() else "None",
        "symptoms": f"Observed symptom pattern corresponding to {disease}.",
        "organic_treatment": "Isolate affected leaves and apply organic copper or neem spray.",
        "chemical_treatment": "Consult local extension specialist for targeted antifungal or antibacterial treatments.",
        "prevention": "Maintain adequate crop spacing, drip irrigation, and pathogen-free seed stock."
    }


def get_data_augmentation_pipeline():
    """
    Returns a Keras Sequential data augmentation pipeline tailored for
    computer vision leaf disease classification.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.1),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        ], name="plant_augmentation_pipeline")
    except ImportError:
        return None


def load_plant_village_dataset(
    dataset_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42
):
    """
    Loads the PlantVillage dataset from a directory into TensorFlow Keras Dataset pipelines
    with automatic train/validation split, batching, caching, and prefetching.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required to load the Keras dataset pipeline. "
            "Install with: pip install tensorflow"
        ) from e

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        class_names=PLANT_VILLAGE_CLASSES if set(os.listdir(dataset_dir)).issuperset(set(PLANT_VILLAGE_CLASSES)) else None
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        class_names=PLANT_VILLAGE_CLASSES if set(os.listdir(dataset_dir)).issuperset(set(PLANT_VILLAGE_CLASSES)) else None
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


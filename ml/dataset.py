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
    "Tomato___healthy",
    # Extended Major Field Crops
    "Wheat___Brown_rust",
    "Wheat___Yellow_rust",
    "Wheat___Septoria",
    "Wheat___healthy",
    "Rice___Brown_spot",
    "Rice___Leaf_blast",
    "Rice___Bacterial_leaf_blight",
    "Rice___healthy",
    "Cotton___Bacterial_blight",
    "Cotton___Leaf_spot",
    "Cotton___Leaf_curl_virus",
    "Cotton___healthy",
    "Sugarcane___Red_rot",
    "Sugarcane___healthy"
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
    },
    # ─── WHEAT ───
    "Wheat___Brown_rust": {
        "plant": "Wheat",
        "display_name": "Wheat - Brown / Leaf Rust (Puccinia triticina)",
        "severity": "Severe",
        "symptoms": "Small, circular to oval, bright orange-brown pustules scattered randomly across the upper leaf blades.",
        "organic_treatment": "Apply bio-fungicides like Trichoderma viride or neem seed kernel extract early.",
        "chemical_treatment": "Spray triazole fungicides (propiconazole 25% EC @ 0.1% or tebuconazole) at early flag leaf stage.",
        "prevention": "Cultivate rust-resistant wheat varieties (e.g. HD-2967, PBW-550) and practice timely sowing."
    },
    "Wheat___Yellow_rust": {
        "plant": "Wheat",
        "display_name": "Wheat - Yellow / Stripe Rust (Puccinia striiformis)",
        "severity": "Severe",
        "symptoms": "Distinct linear parallel stripes of bright yellow-orange pustules aligned along leaf veins.",
        "organic_treatment": "Early field sanitation, remove volunteer grasses, and apply systemic organic bio-fungicides.",
        "chemical_treatment": "Foliar spray of propiconazole 25 EC or azoxystrobin at first sign of yellow stripes.",
        "prevention": "Plant resistant cultivars with Yr genes; monitor during cool, moist weather (10-15°C)."
    },
    "Wheat___Septoria": {
        "plant": "Wheat",
        "display_name": "Wheat - Septoria Leaf Blotch (Zymoseptoria tritici)",
        "severity": "Moderate",
        "symptoms": "Irregular brown necrotic lesions containing tiny black fungal fruiting bodies (pycnidia).",
        "organic_treatment": "Crop rotation with non-gramineous broadleaf crops; compost crop residue.",
        "chemical_treatment": "Apply SDHI fungicides or prothioconazole when flag leaf emerges (GS39).",
        "prevention": "Avoid high seed densities and eradicate volunteer wheat seedlings."
    },
    "Wheat___healthy": {
        "plant": "Wheat",
        "display_name": "Wheat - Healthy Crop Foliage",
        "severity": "None",
        "symptoms": "Vibrant upright green blades, clean tillers, sturdy stems, and no fungal pustules.",
        "organic_treatment": "No treatment needed. Maintain balanced micronutrients (zinc, iron).",
        "chemical_treatment": "None needed.",
        "prevention": "Practice timely sowing and optimal irrigation at crown root initiation."
    },
    # ─── RICE ───
    "Rice___Brown_spot": {
        "plant": "Rice",
        "display_name": "Rice - Brown Spot (Bipolaris oryzae)",
        "severity": "Moderate",
        "symptoms": "Oval to circular dark brown spots with grey or whitish centers across the leaf blades and glumes.",
        "organic_treatment": "Treat seeds with hot water (53-54°C for 10 mins) or Pseudomonas fluorescens.",
        "chemical_treatment": "Foliar spray of propiconazole or iprodione at maximum tillering and panicle initiation.",
        "prevention": "Correct soil potassium and silicon deficiencies; avoid drought stress in nursery beds."
    },
    "Rice___Leaf_blast": {
        "plant": "Rice",
        "display_name": "Rice - Leaf Blast (Magnaporthe oryzae)",
        "severity": "Severe",
        "symptoms": "Spindle-shaped or diamond-shaped lesions with grayish-white centers and dark brown margins.",
        "organic_treatment": "Apply fermented cow dung slurry or silicon foliar sprays to strengthen epidermal cell walls.",
        "chemical_treatment": "Spray tricyclazole 75 WP (0.6 g/L) or isoprothiolane at symptom onset.",
        "prevention": "Avoid excessive nitrogen application; maintain uniform water depth in paddies."
    },
    "Rice___Bacterial_leaf_blight": {
        "plant": "Rice",
        "display_name": "Rice - Bacterial Leaf Blight (Xanthomonas oryzae)",
        "severity": "Severe",
        "symptoms": "Water-soaked to yellowish-white wavy stripes starting from leaf tips and margins, progressing downward.",
        "organic_treatment": "Spray fresh cow dung extract (20%) or copper hydroxide; drain excess standing water.",
        "chemical_treatment": "Apply plantomycin / streptomycin sulfate (100 ppm) combined with copper oxychloride.",
        "prevention": "Use resistant seed varieties; avoid clipping seedling tips at transplanting."
    },
    "Rice___healthy": {
        "plant": "Rice",
        "display_name": "Rice - Healthy Paddy Foliage",
        "severity": "None",
        "symptoms": "Lush emerald-green erect blades, clean sheaths, and vigorous tillering.",
        "organic_treatment": "No treatment needed. Maintain recommended split nitrogen and potash nutrition.",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain proper water regime and avoid stagnant water."
    },
    # ─── COTTON ───
    "Cotton___Bacterial_blight": {
        "plant": "Cotton",
        "display_name": "Cotton - Bacterial Blight / Angular Leaf Spot (Xanthomonas citri pv. malvacearum)",
        "severity": "Severe",
        "symptoms": "Angular water-soaked spots bounded by veins on leaves, turning dark brown/black; blackarm on stems.",
        "organic_treatment": "Delint seeds with sulfuric acid; spray copper-based organic formulations.",
        "chemical_treatment": "Spray copper oxychloride (0.2%) mixed with streptocycline (100 ppm).",
        "prevention": "Use certified acid-delinted seeds and practice deep summer plowing."
    },
    "Cotton___Leaf_spot": {
        "plant": "Cotton",
        "display_name": "Cotton - Alternaria / Cercospora Leaf Spot",
        "severity": "Moderate",
        "symptoms": "Brown circular spots with concentric target rings and reddish-purple borders, causing premature leaf drop.",
        "organic_treatment": "Neem oil foliar application (2%) and balanced potassium supplementation.",
        "chemical_treatment": "Spray mancozeb (2 g/L) or azoxystrobin at early spot development.",
        "prevention": "Avoid moisture stress and ensure adequate potash fertilization."
    },
    "Cotton___Leaf_curl_virus": {
        "plant": "Cotton",
        "display_name": "Cotton - Leaf Curl Virus (CLCuV)",
        "severity": "Severe",
        "symptoms": "Upward or downward cupping of leaves, thickening of veins, and enations (leaf-like outgrowths) on underside.",
        "organic_treatment": "Install yellow sticky traps (10-15 per acre) to control whitefly vector (Bemisia tabaci).",
        "chemical_treatment": "Spray systemic insecticides (diafenthiuron or pymetrozine) to manage whitefly populations.",
        "prevention": "Eradicate weed hosts (Abutilon, Sida) and grow CLCuD-resistant Bt hybrids."
    },
    "Cotton___healthy": {
        "plant": "Cotton",
        "display_name": "Cotton - Healthy Canopy",
        "severity": "None",
        "symptoms": "Broad lobed dark green leaves with clean petioles, healthy squares, and vigorous bolls.",
        "organic_treatment": "No treatment required. Maintain balanced micronutrient foliar spray (boron, magnesium).",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain regular scouting for early sucking pest management."
    },
    # ─── SUGARCANE ───
    "Sugarcane___Red_rot": {
        "plant": "Sugarcane",
        "display_name": "Sugarcane - Red Rot (Colletotrichum falcatum)",
        "severity": "Severe",
        "symptoms": "Discoloration and drying of the spindle leaf; internal stalk tissue turns red with characteristic white horizontal cross-bands.",
        "organic_treatment": "Hot water treatment of seed setts (52°C for 30 minutes); apply Trichoderma harzianum to soil.",
        "chemical_treatment": "Dip setts in carbendazim (0.1%) solution before planting.",
        "prevention": "Adopt crop rotation with rice or green manure; select disease-free setts from healthy seed nurseries."
    },
    "Sugarcane___healthy": {
        "plant": "Sugarcane",
        "display_name": "Sugarcane - Healthy Canopy",
        "severity": "None",
        "symptoms": "Robust elongated dark green leaves with clean midribs and vigorous stalk development.",
        "organic_treatment": "No treatment required. Ensure adequate nitrogen and earthing-up.",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain trash mulching and timely irrigation."
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


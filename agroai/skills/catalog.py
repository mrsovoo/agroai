"""The default AgroAI skill catalogue.

Fifteen skills covering the domain areas of the roadmap: crop production, plant
disease, pest control, soil, irrigation, fertilization, greenhouse, livestock,
veterinary, poultry, animal nutrition, agrochemistry, machinery, Uzbekistan
agriculture and farm management.

Keywords are multilingual (Uzbek, Russian, English) because that is how farmers
actually ask. Guidance strings are short, factual and tool/prompt oriented - a
skill never carries a fact or a dosage that would need a source.
"""

from __future__ import annotations

from agroai.safety.guardrails import HIGH, LOW, MEDIUM
from agroai.skills.base import Skill

DEFAULT_SKILLS: tuple[Skill, ...] = (
    Skill(
        name="crop_production",
        title="Crop production",
        description="Fanning, planting, rotation, growth stages and yields.",
        categories=("crops", "uzbekistan", "machinery"),
        tools=("unit_converter",),
        keywords=(
            "ekish", "ekildi", "hosil", "nav", "bug'doy", "paxta", "makkajo'xori",
            "pomidor", "bodring", "kartoshka", "uzum", "crop", "yield", "planting",
            "rotation", "ekin almashish", "hosildorlik", "посев", "урожай",
        ),
        guidance="Focus on crop-specific agronomy and the growth stage the user mentions.",
    ),
    Skill(
        name="plant_disease",
        title="Plant disease diagnosis",
        description="Symptoms, likely causes and differential reasoning for plant diseases.",
        categories=("plant_diseases", "crops", "greenhouse"),
        tools=("soil_tool", "weather_tool"),
        keywords=(
            "kasallik", "kasalligi", "dog'", "sariq", "jigarrang", "chirish", "zamburug",
            "virus", "bakteriya", "disease", "blight", "mildew", "spot", "wilt", "rot",
            "болезн", "пятн",
        ),
        risk_level=MEDIUM,
        guidance=(
            "List possible causes as hypotheses, ordered by the retrieved evidence, and "
            "say which additional observations would separate them."
        ),
    ),
    Skill(
        name="pest_control",
        title="Pest control",
        description="Insects, mites, nematodes and integrated pest management.",
        categories=("pests", "plant_diseases", "crops"),
        tools=("weather_tool",),
        keywords=(
            "zararkunanda", "shira", "qurt", "kanal", "kanalar", "nematoda", "tuxum",
            "kapalak", "hasharot", "pest", "aphid", "mite", "borer", "ipm",
            "вредител", "тля",
        ),
        risk_level=MEDIUM,
        guidance=(
            "Prefer monitoring and cultural/biological measures first; only cite "
            "chemical options that appear in the retrieved sources."
        ),
    ),
    Skill(
        name="soil_management",
        title="Soil management",
        description="Soil types, pH, EC, salinity, humus, erosion and fertility.",
        categories=("soil", "fertilizers", "irrigation"),
        tools=("soil_tool", "unit_converter"),
        keywords=(
            "tuproq", "sho'rlan", "shorlan", "sho'rin", "ph", "ec", "humus",
            "unumdorlik", "eroziya", "tuz", "saline", "salinity", "soil", "fertility",
            "почва", "засолен",
        ),
        guidance="Use measured soil values from the user; never assume a value.",
    ),
    Skill(
        name="irrigation",
        title="Irrigation and water management",
        description="Irrigation methods, water requirement, drainage and leaching.",
        categories=("irrigation", "soil", "greenhouse"),
        tools=("irrigation_tool", "soil_tool", "unit_converter"),
        keywords=(
            "sug'or", "sugor", "suv", "tomchilat", "yomg'irlat", "drenaj", "namlik",
            "irrigation", "drip", "water", "leaching", "полив", "вода",
        ),
        guidance=(
            "Distinguish the water depth needed from the volume, and state which "
            "measurements (ET0, Kc, ECw) are missing if they are."
        ),
    ),
    Skill(
        name="fertilization",
        title="Plant nutrition and fertilization",
        description="Mineral and organic fertilizers, rates and deficiency symptoms.",
        categories=("fertilizers", "soil", "crops"),
        tools=("calculator", "unit_converter", "soil_tool"),
        keywords=(
            "o'g'it", "og'it", "azot", "fosfor", "kaliy", "mikroelement", "urea",
            "karbamid", "npk", "defitsit", "yetishmovchilik", "fertilizer", "nutrient",
            "удобрен",
        ),
        risk_level=HIGH,
        guidance=(
            "Rates and concentrations must come from the retrieved sources or the "
            "product label; never state a rate from general knowledge."
        ),
    ),
    Skill(
        name="greenhouse",
        title="Greenhouse production",
        description="Microclimate, fertigation, humidity control and greenhouse diseases.",
        categories=("greenhouse", "irrigation", "plant_diseases"),
        tools=("weather_tool", "irrigation_tool"),
        keywords=(
            "issiqxona", "parnik", "teplitsa", "mikroiqlim", "namlik", "fertigation",
            "co2", "greenhouse", "humidity", "теплич",
        ),
        guidance="Separate climate problems from disease problems before recommending action.",
    ),
    Skill(
        name="livestock",
        title="Livestock husbandry",
        description="Cattle, sheep, goats, horses and camels: housing and management.",
        categories=("livestock", "veterinary", "uzbekistan"),
        tools=("veterinary_tool",),
        keywords=(
            "chorva", "qoramol", "sigir", "buzoq", "qo'y", "qoy", "echki", "ot",
            "tuya", "mol", "boqish", "livestock", "cattle", "cow", "sheep", "goat",
            "скот", "коров", "овец",
        ),
        guidance="Deal with husbandry and environment first; clinical questions need a veterinarian.",
    ),
    Skill(
        name="veterinary",
        title="Veterinary medicine",
        description="Animal diseases, symptoms, prevention and vaccination programmes.",
        categories=("veterinary", "livestock", "poultry", "medicines"),
        tools=("veterinary_tool",),
        keywords=(
            "veterinar", "kasallik belgilari", "harorat", "isitma", "yotal", "yutal",
            "ich ket", "qon", "emlash", "vaksina", "antibiotik", "to'ng'iz", "symptom",
            "disease", "fever", "vaccination", "antibiotic", "ветеринар", "лечен",
        ),
        risk_level=HIGH,
        guidance=(
            "Structure the case (observed signs, required observations, urgency) and "
            "recommend a veterinarian; never diagnose or dose from memory."
        ),
    ),
    Skill(
        name="poultry",
        title="Poultry production",
        description="Broilers, layers, turkeys and ducks: management and health.",
        categories=("poultry", "veterinary", "livestock"),
        tools=("veterinary_tool",),
        keywords=(
            "parranda", "tovuq", "jo'ja", "broiler", "tuxum", "kurka", "o'rdak",
            "poultry", "chicken", "layer", "duck", "птиц", "куриц",
        ),
        risk_level=HIGH,
        guidance="Poultry disease spreads fast: mention biosecurity and prompt veterinary contact.",
    ),
    Skill(
        name="animal_nutrition",
        title="Animal nutrition",
        description="Feed rations, feed quality, water and deficiency problems.",
        categories=("livestock", "poultry", "veterinary"),
        tools=("calculator", "unit_converter"),
        keywords=(
            "yem", "ozuqa", "ratsion", "korm", "yem-xashak", "oziqa", "feed", "ration",
            "fodder", "корм",
        ),
        guidance="Ask for the animal's weight, age and production level before any ration arithmetic.",
    ),
    Skill(
        name="agrochemistry",
        title="Agrochemicals and plant protection products",
        description="Fungicides, insecticides, herbicides and their safe use.",
        categories=("medicines", "fertilizers", "pests"),
        tools=("calculator", "weather_tool"),
        keywords=(
            "pestitsid", "fungitsid", "gerbitsid", "insektitsid", "zahar", "purkash",
            "doza", "konsentratsiya", "kutish muddati", "pesticide", "fungicide",
            "herbicide", "spray", "dose", "пестицид", "доза",
        ),
        risk_level=HIGH,
        guidance=(
            "The label outranks everything: cite only retrieved rates, mention the "
            "pre-harvest interval and protective equipment."
        ),
    ),
    Skill(
        name="machinery",
        title="Agricultural machinery and technology",
        description="Machinery, sensors, drones, precision agriculture and IoT.",
        categories=("machinery", "crops"),
        tools=("unit_converter", "calculator"),
        keywords=(
            "texnika", "traktor", "kombayn", "urug' seyalka", "sensor", "dron",
            "iot", "precision", "machine", "tractor", "drone", "техник", "трактор",
        ),
        guidance="Keep machinery answers practical and maintenance-oriented.",
    ),
    Skill(
        name="uzbekistan_agriculture",
        title="Uzbekistan context",
        description="Regions, climate, water and practice specific to Uzbekistan.",
        categories=("uzbekistan", "crops", "livestock", "irrigation"),
        tools=("soil_tool", "irrigation_tool", "weather_tool"),
        keywords=(
            "o'zbekiston", "uzbekistan", "sirdaryo", "samarqand", "buxoro", "xorazm",
            "farg'ona", "andijon", "namangan", "qashqadaryo", "surxondaryo", "navoiy",
            "jizzax", "toshkent", "qoraqalpog'iston", "viloyat", "region",
        ),
        guidance=(
            "Prefer region-specific retrieved material and say clearly when the "
            "knowledge base has nothing for that region."
        ),
    ),
    Skill(
        name="farm_management",
        title="Farm management",
        description="Planning, field records, costs and decision support.",
        categories=("crops", "livestock", "uzbekistan"),
        tools=("calculator", "unit_converter"),
        keywords=(
            "xo'jalik", "ferma", "reja", "rejalashtir", "hisob", "xarajat", "foyda",
            "management", "plan", "budget", "cost", "планирован", "хозяйств",
        ),
        guidance="Show the arithmetic behind any planning figure so the farmer can check it.",
    ),
)

__all__ = ["DEFAULT_SKILLS"]

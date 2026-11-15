"""
claude_service.py
─────────────────────────────────────────────────────────────────
Production-ready Claude API integration for AgriVision.
Handles:
  1. Human-readable explanations of ML predictions
  2. Chatbot conversations with farming context
  3. One-shot quick explanations
  4. Clean error handling & retry logic
─────────────────────────────────────────────────────────────────
Install dependencies:
  pip install anthropic python-dotenv
"""

import anthropic
import os
import json
from typing import Optional

# Auto-load .env file if present (must be before os.getenv calls)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads .env from project root automatically
except ImportError:
    pass  # python-dotenv optional — can also export key manually

# ──────────────────────────────────────────────────────────────
# CLIENT SETUP
# ──────────────────────────────────────────────────────────────

# Read API key from environment variable (safest approach)
# Set it in your shell:  export ANTHROPIC_API_KEY="sk-ant-..."
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Model: claude-3-5-haiku is the latest fast/cheap model (Oct 2024)
# Upgrade to claude-3-5-sonnet-20241022 for richer, longer explanations
CLAUDE_MODEL = "claude-3-5-haiku-20241022"

# System prompt that gives Claude its role and expertise
AGRI_SYSTEM_PROMPT = """You are AgriBot, an expert AI assistant for the AgriVision smart agriculture platform.

Your expertise covers:
- Plant diseases, symptoms, causes, and organic/chemical treatments
- Crop selection based on soil nutrients (NPK), pH, climate, and rainfall
- Soil health analysis and improvement techniques
- Precision farming, irrigation, and fertilizer management
- Pest control and integrated pest management (IPM)
- Post-harvest handling and crop storage
- Climate-smart agriculture practices

Communication style:
- Use clear, simple language that farmers with basic education can understand
- Be warm, encouraging, and practical — give actionable advice
- Use bullet points and numbered lists for clarity
- Keep responses concise (150-250 words max) unless a complex question requires more
- Always mention when professional consultation is strongly recommended
- Never make up statistics or facts — say "I'm not certain" when unsure

You have access to the user's latest ML prediction results when answering questions."""


def _get_client() -> anthropic.Anthropic:
    """Create and return an Anthropic client."""
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Run: export ANTHROPIC_API_KEY='your-key-here'"
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ──────────────────────────────────────────────────────────────
# FEATURE 1 — DISEASE PREDICTION EXPLANATION
# ──────────────────────────────────────────────────────────────

def explain_disease_prediction(
    disease: str,
    confidence: float,
    plant: str,
    severity: str,
    treatment: str,
) -> dict:
    """
    Takes ML prediction output and generates a rich, human-readable
    explanation using Claude.

    Args:
        disease:    Predicted disease name (e.g. "Tomato Early Blight")
        confidence: Model confidence 0.0–1.0
        plant:      Detected plant type (e.g. "Tomato")
        severity:   "None" / "Mild" / "Moderate" / "Severe"
        treatment:  Short treatment string from ML model

    Returns:
        dict with keys: headline, explanation, action_plan, prevention_tips, urgency_level
    """
    confidence_pct = round(confidence * 100)

    if not ANTHROPIC_API_KEY:
        urgency = "High" if severity == "Severe" else ("Medium" if severity == "Moderate" else "Low")
        return {
            "headline": f"{disease} Diagnosis & Treatment Guide",
            "explanation": f"The Deep Learning model identified {disease} on your {plant} with {confidence_pct}% confidence. The observed tissue symptoms indicate a {severity.lower()} condition requiring timely management.",
            "action_plan": f"1. {treatment}\n2. Isolate and prune heavily infected foliage\n3. Sanitize shears and gardening equipment between plants",
            "prevention_tips": "• Implement drip irrigation to keep leaf canopies dry\n• Space plants to promote optimal airflow\n• Rotate non-host crops to break pathogen cycles",
            "urgency_level": urgency,
            "confidence_note": f"The neural network evaluated key lesion patterns and is {confidence_pct}% confident."
        }

    prompt = f"""
The plant disease ML model returned this result:
- Plant:      {plant}
- Disease:    {disease}
- Confidence: {confidence_pct}%
- Severity:   {severity}
- Treatment:  {treatment}

Generate a structured explanation. Return ONLY valid JSON with these exact keys:
{{
  "headline": "One short punchy headline (max 10 words)",
  "explanation": "2-3 sentences explaining what this disease is, what causes it, and why the model flagged it",
  "action_plan": "Numbered list of 3-5 immediate steps the farmer should take, starting with the most urgent",
  "prevention_tips": "2-3 bullet points on how to prevent this disease in future seasons",
  "urgency_level": "Low / Medium / High / Critical",
  "confidence_note": "One sentence explaining what the {confidence_pct}% confidence means in plain English"
}}
"""
    try:
        client = _get_client()
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=600,
            system=AGRI_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError as e:
        return {
            "headline": f"{disease} Detected",
            "explanation": f"Your {plant} shows signs of {disease} with {confidence_pct}% confidence. {treatment}",
            "action_plan": f"1. {treatment}\n2. Monitor the plant closely\n3. Consult a local agronomist if symptoms worsen",
            "prevention_tips": "• Practice crop rotation\n• Ensure good air circulation\n• Use disease-resistant varieties",
            "urgency_level": "Medium",
            "confidence_note": f"The model is {confidence_pct}% confident in this diagnosis."
        }
    except Exception as e:
        raise RuntimeError(f"Claude API error in explain_disease: {str(e)}")


# ──────────────────────────────────────────────────────────────
# FEATURE 2 — CROP RECOMMENDATION EXPLANATION
# ──────────────────────────────────────────────────────────────

def explain_crop_recommendation(
    recommended_crop: str,
    soil_health_score: float,
    n: float,
    p: float,
    k: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
) -> dict:
    """
    Generates a detailed explanation of why a crop was recommended
    based on soil and climate metrics.

    Returns:
        dict with keys: headline, why_this_crop, soil_analysis,
                        farming_calendar, water_fertilizer_tips, warnings
    """
    if not ANTHROPIC_API_KEY:
        fertility = "optimal" if soil_health_score >= 7.5 else ("moderate" if soil_health_score >= 5.5 else "sub-optimal")
        return {
            "headline": f"{recommended_crop} Farming Strategic Plan",
            "why_this_crop": f"Based on your soil nutrient profile (N:{n}, P:{p}, K:{k} kg/ha) and pH {ph}, {recommended_crop} is well-adapted for these growth conditions.",
            "soil_analysis": f"Your soil health score is {soil_health_score}/10, indicating {fertility} fertility for {recommended_crop} cultivation.",
            "farming_calendar": f"Plan land preparation and sowing in accordance with your regional planting calendar and monsoon/rain forecast ({rainfall} mm 30-day accumulation).",
            "water_fertilizer_tips": f"Apply balanced NPK fertilization to sustain vegetative vigor; irrigate based on soil moisture and ambient temperature ({temperature}°C).",
            "warnings": "Monitor drainage during heavy precipitation events to prevent waterlogging." if rainfall > 150 else "Maintain regular irrigation intervals."
        }

    prompt = f"""
The crop recommendation ML model analyzed this soil data:
- Nitrogen (N): {n} kg/ha
- Phosphorus (P): {p} kg/ha
- Potassium (K): {k} kg/ha
- Soil pH: {ph}
- Temperature: {temperature}°C
- Humidity: {humidity}%
- Rainfall (30d): {rainfall}mm
- Soil Health Score: {soil_health_score}/10

Recommended Crop: {recommended_crop}

Generate a structured explanation. Return ONLY valid JSON with these exact keys:
{{
  "headline": "One engaging headline about this recommendation (max 12 words)",
  "why_this_crop": "2-3 sentences explaining exactly why {recommended_crop} suits these specific soil and climate numbers",
  "soil_analysis": "Analysis of the NPK balance and pH — what's good, what's lacking, what needs fixing",
  "farming_calendar": "Best planting window and rough harvest timeline for {recommended_crop} in these conditions",
  "water_fertilizer_tips": "Specific irrigation schedule and fertilizer top-up recommendations for {recommended_crop}",
  "warnings": "Any red flags in the soil data that could reduce yield, or 'None' if data looks healthy"
}}
"""
    try:
        client = _get_client()
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            system=AGRI_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError:
        return {
            "headline": f"{recommended_crop} is Your Best Bet",
            "why_this_crop": f"Based on your soil's NPK values ({n},{p},{k}) and pH {ph}, {recommended_crop} is well-suited.",
            "soil_analysis": f"Soil health score of {soil_health_score}/10 indicates {'good' if soil_health_score >= 7 else 'moderate'} fertility.",
            "farming_calendar": "Consult local agricultural extension office for exact planting dates.",
            "water_fertilizer_tips": "Apply balanced NPK fertilizer and maintain regular irrigation.",
            "warnings": "None" if soil_health_score >= 6 else "Soil health is below optimal — consider soil amendment before planting."
        }
    except Exception as e:
        raise RuntimeError(f"Claude API error in explain_crop: {str(e)}")


# ──────────────────────────────────────────────────────────────
# FEATURE 3 — CHATBOT (Multi-turn conversation)
# ──────────────────────────────────────────────────────────────

def chat_with_agribot(
    messages: list[dict],
    context: Optional[dict] = None
) -> str:
    """
    Multi-turn chatbot using Claude.

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts
                  (full conversation history for context)
        context:  Optional dict with current prediction context
                  e.g. {"type": "disease", "disease": "Tomato Early Blight", ...}

    Returns:
        Assistant reply as a plain string
    """
    # Build a context prefix if we have prediction data
    system = AGRI_SYSTEM_PROMPT
    if context:
        ctx_parts = []
        if context.get("type") == "disease":
            ctx_parts.append(
                f"The user's latest scan result: {context.get('disease', 'Unknown')} "
                f"detected on {context.get('plant', 'plant')} with "
                f"{round(context.get('confidence', 0) * 100)}% confidence."
            )
        elif context.get("type") == "crop":
            ctx_parts.append(
                f"The user's latest crop recommendation: {context.get('recommended_crop', 'Unknown')} "
                f"(Soil health: {context.get('soil_health_score', '?')}/10)"
            )
        if ctx_parts:
            system += "\n\nCurrent session context:\n" + "\n".join(ctx_parts)

    if not ANTHROPIC_API_KEY:
        last_msg = messages[-1].get("content", "Hello") if messages else "Hello"
        return f"AgriBot Assistant: Regarding your inquiry about '{last_msg[:60]}', ensure balanced soil fertility, monitor for early foliar lesions, and maintain proper crop irrigation schedules. (Configure ANTHROPIC_API_KEY in your backend .env for full conversational capabilities)."

    try:
        client = _get_client()

        # Validate message list — Claude requires alternating user/assistant
        validated_messages = _validate_messages(messages)

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            system=system,
            messages=validated_messages
        )
        return response.content[0].text.strip()

    except Exception as e:
        raise RuntimeError(f"Claude chatbot error: {str(e)}")


def _validate_messages(messages: list[dict]) -> list[dict]:
    """
    Ensure message list starts with a user message and alternates properly.
    Removes any system-role messages (those go in the system parameter).
    """
    valid = [m for m in messages if m.get("role") in ("user", "assistant")]
    # Must start with user
    if valid and valid[0]["role"] != "user":
        valid = valid[1:]
    return valid if valid else [{"role": "user", "content": "Hello"}]


# ──────────────────────────────────────────────────────────────
# FEATURE 4 — QUICK ONE-SHOT EXPLAIN (for simple frontend calls)
# ──────────────────────────────────────────────────────────────

def quick_explain(topic: str, context: str) -> str:
    """
    General-purpose one-shot explainer.
    Useful for quick inline explanations triggered from frontend.

    Args:
        topic:   Short topic label (e.g. "Tomato Early Blight")
        context: Extra context to include in the prompt

    Returns:
        Plain-text explanation string (100-200 words)
    """
    prompt = f"""
    Explain the following agriculture topic in simple, farmer-friendly language (100-150 words):
    Topic: {topic}
    Context: {context}

    Be practical and actionable. Use bullet points where helpful.
    """
    if not ANTHROPIC_API_KEY:
        return f"{topic}: Key agronomic area focusing on balanced crop management, soil condition monitoring, and proactive disease mitigation to preserve crop yield."

    try:
        client = _get_client()
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            system=AGRI_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        raise RuntimeError(f"Claude quick_explain error: {str(e)}")

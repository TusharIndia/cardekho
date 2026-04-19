import os
import json
import re
import pandas as pd
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("gemini_api_key")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
else:
    model = None

# Load cleaned car dataset
df = pd.read_csv("cleaned_car_dataset.csv")
KNOWN_MAKES = sorted(df["make"].dropna().unique().tolist(), key=len, reverse=True)
KNOWN_MODELS = sorted(df["model"].dropna().unique().tolist(), key=len, reverse=True)
GENERIC_MODEL_TOKENS = {
    "edition", "turbo", "sport", "line", "plus", "smart", "premium", "automatic",
    "manual", "petrol", "diesel", "electric", "hybrid", "variant", "standard"
}
KNOWN_MODEL_TOKENS = sorted(
    {
        token
        for model in KNOWN_MODELS
        for token in re.findall(r"[a-z0-9]+", model.lower())
        if len(token) >= 4 and token not in GENERIC_MODEL_TOKENS
    },
    key=len,
    reverse=True,
)

LARGE_FAMILY_MODEL_KEYWORDS = {
    "ertiga", "xl6", "invicto", "carens", "alcazar", "safari", "xuv700",
    "scorpio", "innova", "rumion", "hector plus", "bolero neo plus", "triber"
}


def _to_rupees(value, unit):
    unit = (unit or "").lower()
    if unit in {"crore", "cr"}:
        return int(float(value) * 10000000)
    if unit in {"lakh", "lac", "l"}:
        return int(float(value) * 100000)
    if unit in {"k", "thousand"}:
        return int(float(value) * 1000)
    return int(float(value))


def _extract_budget_from_query(query):
    text = query.lower().replace(",", "")

    budget_patterns = [
        r'(?:under|below|less than|upto|up to|within|not more than|no more than|at most|budget(?:\s+of)?|around|about|max(?:imum)?|spend|cost|price(?:\s+of)?)\s*₹?\s*(\d+(?:\.\d+)?)\s*(crore|cr|lakh|lac|l|k|thousand)?',
        r'₹\s*(\d+(?:\.\d+)?)\s*(crore|cr|lakh|lac|l|k|thousand)?',
        r'(\d+(?:\.\d+)?)\s*(crore|cr|lakh|lac|l|k|thousand)\b',
        r'(?:under|below|less than|upto|up to|within|budget(?:\s+of)?|around|about|max(?:imum)?)\s*₹?\s*(\d+(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|thousand)?',
        r'(?:budget|spend|cost|price)\s*(?:is|=|:)?\s*(\d{5,9})(?!\s*(?:km|kmpl|bhp|stars?))',
    ]

    for pattern in budget_patterns:
        match = re.search(pattern, text)
        if match:
            amount = match.group(1)
            unit = match.group(2) if len(match.groups()) > 1 else ""
            return _to_rupees(amount, unit)

    # Fallback: "my budget is 500000" style with no explicit unit.
    raw_num = re.search(r'\b(?:budget|spend|cost|price)\b[^\d]{0,12}(\d{5,9})\b', text)
    if raw_num:
        return int(raw_num.group(1))

    return None


def _extract_family_size(query):
    text = query.lower()
    match = re.search(r'family(?:\s+size)?\s*(?:of)?\s*(\d+)', text)
    if match:
        return max(1, int(match.group(1)))

    match = re.search(r'family\s*(?:is|=|:)?\s*(\d+)', text)
    if match:
        return max(1, int(match.group(1)))

    match = re.search(r'(\d+)\s*(?:members|people|persons)', text)
    if match:
        return max(1, int(match.group(1)))

    match = re.search(r'\b(\d+)\s*-?\s*seater\b', text)
    if match:
        return max(1, int(match.group(1)))

    if "single" in text:
        return 1
    if any(word in text for word in ["couple", "2 people", "two people"]):
        return 2
    if any(word in text for word in ["large family", "big family"]):
        return 6
    return None


def _extract_preference(query):
    text = query.lower()
    if any(word in text for word in ["safety", "safe", "airbag", "airbags", "adas", "crash", "secure"]):
        return "safety"
    if any(word in text for word in ["performance", "power", "fast", "speed", "horsepower", "bhp", "luxury", "premium", "sporty", "executive"]):
        return "performance"
    if any(word in text for word in ["mileage", "efficiency", "fuel", "economy", "economical"]):
        return "efficiency"
    if any(word in text for word in ["budget", "cheap", "affordable", "low cost", "cheaper", "value"]):
        return "budget"
    return None


def _is_strict_budget_query(query):
    """Detect queries that require a hard upper budget cap (no flex)."""
    text = query.lower()
    strict_markers = [
        "under",
        "below",
        "less than",
        "lower than",
        "within",
        "upto",
        "up to",
        "not more than",
        "no more than",
        "at most",
        "maximum",
        "max",
        "ke andar",
    ]
    return any(marker in text for marker in strict_markers)


def parse_natural_language_query(query):
    """Parse natural language query with robust rule-based fallback + optional Gemini refinement."""
    extracted_budget = _extract_budget_from_query(query)
    extracted_family = _extract_family_size(query)
    extracted_preference = _extract_preference(query)

    parsed = {
        "budget": extracted_budget if extracted_budget is not None else 1000000,
        "family_size": extracted_family if extracted_family is not None else 3,
        "preference": extracted_preference if extracted_preference is not None else "balanced",
        "provided_budget": extracted_budget is not None,
        "provided_family_size": extracted_family is not None,
        "provided_preference": extracted_preference is not None,
        "raw_query": query,
    }

    if not model:
        return parsed

    try:
        prompt = (
            'Parse this car query and return ONLY valid JSON with keys: '
            'budget (integer rupees), family_size (integer), '
            'preference (budget/performance/safety/efficiency/balanced), raw_query. '
            f'Query: "{query}"'
        )
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1].replace("json\n", "").strip()

        gemini_parsed = json.loads(text)

        gemini_budget = gemini_parsed.get("budget")
        if extracted_budget is None and isinstance(gemini_budget, (int, float)) and gemini_budget > 0:
            parsed["budget"] = int(gemini_budget)
            parsed["provided_budget"] = True

        gemini_family = gemini_parsed.get("family_size")
        if extracted_family is None and isinstance(gemini_family, (int, float)) and gemini_family > 0:
            parsed["family_size"] = int(gemini_family)
            parsed["provided_family_size"] = True

        gemini_pref = str(gemini_parsed.get("preference", "")).lower().strip()
        if extracted_preference is None and gemini_pref in {"budget", "performance", "safety", "efficiency", "balanced"}:
            parsed["preference"] = gemini_pref
            parsed["provided_preference"] = gemini_pref != "balanced"

        parsed["raw_query"] = query
        return parsed
    except Exception:
        return parsed


def generate_car_suggestion(car, budget, family_size, preference):
    """Generate personalized suggestion for a car."""
    if not model:
        return "Perfect fit!"
    try:
        prompt = f"Suggest why {car['make']} {car['model']} (₹{car['price']:,}, {car['mileage']} kmpl, {car['safety_rating']}/5 safety) is perfect for: Budget ₹{budget:,}, family {family_size}, {preference} cars. Keep to 1-2 sentences, friendly."
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return "Great choice for your needs!"


def generate_fallback_suggestion(budget, family_size, preference):
    """Generate helpful suggestion when no cars found."""
    if not model:
        return "Try adjusting budget or preferences to find more options!"
    try:
        cars_just_over = df[(df["price"] > budget) & (df["price"] <= budget * 1.3)]
        alt = ""
        if len(cars_just_over) > 0:
            c = cars_just_over.iloc[0]
            alt = f" Try {c['make']} {c['model']} by adding ₹{int(c['price']-budget):,} more."
        prompt = f"User searched for cars with budget ₹{budget:,}, family {family_size}, preference {preference}. No exact match.{alt} Give 2-3 sentence encouraging suggestion with smart budget adjustment or trade-off ideas."
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return "Try adjusting your budget slightly or preferences to find great options!"


def is_car_related_query(query):
    """Check if query is related to cars or car recommendations."""
    if not query:
        return True
    
    query_lower = query.lower()

    non_car_keywords = [
        "weather", "joke", "laptop", "python", "sql", "recipe", "movie", "phone", "table",
        "coding", "program", "football", "cricket"
    ]

    strong_car_keywords = [
        "car", "vehicle", "automobile", "sedan", "suv", "hatchback", "mpv", "ev", "electric",
        "diesel", "petrol", "cng", "mileage", "safety", "bhp", "power", "seating", "seater",
        "transmission", "highway", "commute", "family", "performance", "rating"
    ]

    budget_intent_keywords = [
        "budget", "lakh", "lac", "crore", "rs", "rupees", "price", "cost", "spend",
        "cheap", "affordable", "cheaper", "value", "under", "below", "within"
    ]

    has_make = any(make.lower() in query_lower for make in KNOWN_MAKES)
    has_model_full = any(model.lower() in query_lower for model in KNOWN_MODELS if len(model) >= 4)
    has_model_token = any(re.search(rf"\b{re.escape(tok)}\b", query_lower) for tok in KNOWN_MODEL_TOKENS)
    has_strong_car = any(keyword in query_lower for keyword in strong_car_keywords) or has_make or has_model_full or has_model_token

    # If query clearly belongs to another domain and has no strong car signal, mark as non-car.
    if any(keyword in query_lower for keyword in non_car_keywords) and not has_strong_car:
        return False

    if has_strong_car:
        return True

    # Budget-only and follow-up style intents are treated as car-related in this chatbot.
    if any(keyword in query_lower for keyword in budget_intent_keywords):
        return True

    followup_markers = ["another option", "same budget", "safer", "better mileage", "this range", "current option"]
    if any(marker in query_lower for marker in followup_markers):
        return True

    return False


def _extract_make_from_query(query):
    query_lower = query.lower()
    for make in KNOWN_MAKES:
        if make.lower() in query_lower:
            return make
    return None


def _estimate_family_size_bucket(row):
    """Estimate family-size suitability from model/variant cues available in dataset."""
    text = f"{str(row.get('model', ''))} {str(row.get('variant', ''))}".lower()

    # Explicit seat cues in variant/model text if present
    if re.search(r'\b(8|9|10)\s*-?\s*seater\b', text):
        return "8+"
    if re.search(r'\b(6|7)\s*-?\s*seater\b', text):
        return "6-7"

    # Known large-family model names
    if any(keyword in text for keyword in LARGE_FAMILY_MODEL_KEYWORDS):
        return "6-7"

    # Default dataset-level estimate
    return "3-5"


def _summarize_family_size_distribution(rows_df):
    buckets = {"3-5": 0, "6-7": 0, "8+": 0}
    for _, row in rows_df.iterrows():
        buckets[_estimate_family_size_bucket(row)] += 1

    non_zero = [k for k, v in buckets.items() if v > 0]
    if not non_zero:
        return "3-5", buckets

    primary = max(buckets, key=buckets.get)
    return primary, buckets


def _translate_info_query(query):
    """Translate informational query into structured intent using Gemini when available."""
    fallback = {
        "intent": "seating_capacity" if _is_seating_capacity_query(query) else "unknown",
        "make": _extract_make_from_query(query),
    }

    if not model:
        return fallback

    try:
        prompt = (
            "Extract intent and make from this car-info query. Return ONLY JSON with keys: "
            "intent (seating_capacity or unknown), make (string or empty). "
            f"Query: \"{query}\""
        )
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1].replace("json\n", "").strip()

        parsed = json.loads(text)
        intent = str(parsed.get("intent", "unknown")).strip().lower()
        make = str(parsed.get("make", "")).strip()

        if intent not in {"seating_capacity", "unknown"}:
            intent = fallback["intent"]

        if not make:
            make = fallback["make"]

        # Normalize make to dataset value if possible
        if make:
            exact = next((m for m in KNOWN_MAKES if m.lower() == make.lower()), None)
            if exact:
                make = exact
            else:
                contains = next((m for m in KNOWN_MAKES if make.lower() in m.lower() or m.lower() in make.lower()), None)
                make = contains or fallback["make"]

        return {"intent": intent, "make": make}
    except Exception:
        return fallback


def _is_seating_capacity_query(query):
    q = query.lower()
    patterns = [
        "seating capacity",
        "seat capacity",
        "how many seats",
        "seater",
        "seats",
    ]
    return any(p in q for p in patterns)


def _handle_dataset_info_query(query):
    """Handle informational dataset queries that are not recommendation-style prompts."""
    translated = _translate_info_query(query)
    if translated.get("intent") != "seating_capacity":
        return None

    make = translated.get("make")
    if make:
        subset = df[df["make"].str.lower() == make.lower()].copy()
    else:
        subset = df.copy()

    if subset.empty:
        return {
            "success": False,
            "error": "I could not find that brand in the dataset.",
            "recommendations": [],
        }, 400

    # Dataset has no seating-capacity column; provide estimated family-size suitability.
    primary_bucket, bucket_counts = _summarize_family_size_distribution(subset)

    top = subset.sort_values(["user_rating", "safety_rating"], ascending=False).head(5)
    recs = top.to_dict(orient="records")

    for rec in recs:
        est_bucket = _estimate_family_size_bucket(rec)
        rec["suggestion"] = f"Estimated family-size suitability: {est_bucket}"

    if make:
        message = (
            f"For {make}, exact seating-capacity data is not present in this dataset. "
            f"Based on model/variant cues, estimated suitability is mainly for family size {primary_bucket}. "
            f"Showing top-rated {make} options with estimated family-size suitability."
        )
    else:
        message = (
            "Exact seating-capacity data is not present in this dataset. "
            f"Estimated suitability across available cars is mainly family size {primary_bucket}. "
            "Showing top-rated options with estimated family-size suitability."
        )

    # Optional quick distribution detail for transparency
    dist_parts = [f"{k}: {v}" for k, v in bucket_counts.items() if v > 0]
    if dist_parts:
        message += " Distribution -> " + ", ".join(dist_parts)

    return {
        "success": True,
        "filter_reason": "Dataset information response",
        "recommendations": recs,
        "gemini_summary": "",
        "count": len(recs),
        "user_message": f"✅ {message}",
    }, 200


def get_minimum_budget_car():
    """Get the car with minimum price."""
    if len(df) == 0:
        return None
    min_car = df.loc[df["price"].idxmin()]
    return min_car


def filter_cars(budget_rupees, family_size, preference, strict_budget=False):
    """
    Rule-based car filtering engine with flexible budget.
    
    Args:
        budget_rupees: Max budget in rupees
        family_size: 1-2 (compact), 3-5 (sedan/compact suv), 5+ (suv/mpv)
        preference: 'budget', 'mid', 'premium', 'performance', 'safety', 'efficiency'
    
    Returns:
        Filtered DataFrame, recommendation reason, and personalized suggestions
    """
    filtered = df.copy()
    reasons = []

    # Use hard cap for strict queries (e.g. "under 4 lakh"), otherwise allow 15% flex
    budget_ceiling = budget_rupees if strict_budget else budget_rupees * 1.15
    filtered = filtered[filtered["price"] <= budget_ceiling]
    if strict_budget:
        reasons.append(f"Strictly within ₹{budget_rupees:,} budget")
    else:
        reasons.append(f"Within ₹{budget_rupees:,} budget range")

    # Family size heuristic
    if family_size <= 2:
        # Prefer compact cars with decent mileage
        filtered["score"] = (
            filtered["mileage"] * 0.4 + filtered["power"] * 0.3 + filtered["safety_rating"] * 0.3
        )
        reasons.append("Compact car optimized")
    elif family_size <= 5:
        # Balanced across safety, comfort (power for acceleration)
        filtered["score"] = (
            filtered["safety_rating"] * 0.4 + filtered["power"] * 0.3 + filtered["mileage"] * 0.3
        )
        reasons.append("Comfort and safety optimized")
    elif family_size <= 7:
        # 6-7 members: stronger focus on safety and performance for loaded travel
        filtered["score"] = (
            filtered["safety_rating"] * 0.5 + filtered["power"] * 0.35 + filtered["mileage"] * 0.15
        )
        reasons.append("Large family (6-7) optimized")
    else:
        # 8+ members: dataset has no seat-count column, so apply stricter safety/power heuristic
        filtered = filtered[(filtered["safety_rating"] >= 4.0) & (filtered["power"] >= 90)]
        filtered["score"] = (
            filtered["safety_rating"] * 0.6 + filtered["power"] * 0.4
        )
        reasons.append("Very large family (8+) strict safety/power shortlist")

    # Preference-based scoring adjustments
    if preference == "budget":
        filtered["score"] += filtered["mileage"] * 0.2  # Boost fuel efficiency
        reasons.append("Budget-conscious optimization")
    elif preference == "performance":
        filtered["score"] += filtered["power"] * 0.3  # Boost power
        reasons.append("Performance-focused selection")
    elif preference == "safety":
        filtered["score"] += filtered["safety_rating"] * 0.3  # Boost safety
        reasons.append("Safety-prioritized choice")
    elif preference == "efficiency":
        filtered["score"] += filtered["mileage"] * 0.3  # Boost mileage
        reasons.append("Fuel efficiency optimized")

    # Sort by score and get top 5
    filtered = filtered.sort_values("score", ascending=False).head(5)

    if len(filtered) == 0:
        return pd.DataFrame(), " | ".join(reasons), []

    filtered = filtered.drop("score", axis=1).reset_index(drop=True)
    
    # Generate personalized suggestions for each car
    suggestions = []
    for idx, row in filtered.iterrows():
        suggestion = generate_car_suggestion(row.to_dict(), budget_rupees, family_size, preference)
        suggestions.append(suggestion)
    
    return filtered, " | ".join(reasons), suggestions


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    
    # Check if user provided natural language query
    query = data.get("query", "").strip()
    irrelevant_query = False
    
    if query:
        # Check if query is related to cars
        if not is_car_related_query(query):
            irrelevant_query = True
            return jsonify({
                "success": False,
                "error": "Sorry, I can only help with car recommendations!",
                "irrelevant_query": True,
                "fallback_message": "But I can still help you find a great car! Let me recommend some excellent options based on a typical budget, family size, and preference.",
                "recommendations": [],
                "user_message": None
            }), 400

        # Handle dataset-information questions (e.g., seating capacity) directly
        info_response, status_code = _handle_dataset_info_query(query) or (None, None)
        if info_response is not None:
            return jsonify(info_response), status_code
        
        # Parse natural language query with Gemini
        parsed = parse_natural_language_query(query)
        budget = parsed.get("budget", 1000000)
        family_size = parsed.get("family_size", 3)
        preference = parsed.get("preference", "balanced")
        strict_budget = _is_strict_budget_query(query)

        understood_parts = []
        if parsed.get("provided_budget"):
            understood_parts.append(f"Budget ₹{budget:,}")
        if parsed.get("provided_family_size"):
            understood_parts.append(f"family of {family_size}")
        if parsed.get("provided_preference"):
            understood_parts.append(f"{preference} preference")

        if understood_parts:
            user_message = "✅ Understood: " + ", ".join(understood_parts)
        else:
            user_message = "✅ Understood your request. I will suggest the best matching options."

        if family_size >= 8:
            user_message += " Note: For very large families, exact seat-count data is not available in this dataset, so I am prioritizing higher safety and power options."
    else:
        # Use form input
        budget = data.get("budget", 1000000)
        family_size = data.get("family_size", 3)
        preference = data.get("preference", "balanced")
        strict_budget = False
        user_message = None

    # Check if budget is below the minimum car price in dataset
    min_car = get_minimum_budget_car()
    if min_car is not None and budget < min_car["price"]:
        min_price = int(min_car["price"])
        return jsonify({
            "success": False,
            "error": "Budget too low",
            "budget_too_low": True,
            "min_budget_message": f"Sorry, we don't have any car under ₹{int(budget):,}. Our minimum available car starts at ₹{min_price:,}. I can recommend that option for you.",
            "min_budget_car": {
                "make": min_car["make"],
                "model": min_car["model"],
                "price": min_price,
                "price_display": f"₹{min_price:,}",
                "mileage": min_car["mileage"],
                "power": min_car["power"],
                "safety_rating": min_car["safety_rating"],
                "user_rating": min_car["user_rating"]
            },
            "user_message": user_message
        }), 400

    # Get filtered recommendations and suggestions
    recommendations, reason, suggestions = filter_cars(budget, family_size, preference, strict_budget=strict_budget)

    if recommendations.empty:
        # Generate helpful fallback suggestion
        fallback = generate_fallback_suggestion(budget, family_size, preference)
        return jsonify({
            "success": False,
            "error": "No exact matches found",
            "filter_reason": reason,
            "fallback_suggestion": fallback,
            "recommendations": [],
            "user_message": user_message
        }), 400

    # Convert to list of dicts for JSON response
    recs_list = recommendations.to_dict(orient="records")
    
    # Add personalized suggestions to each recommendation
    for i, rec in enumerate(recs_list):
        rec["suggestion"] = suggestions[i] if i < len(suggestions) else "Great choice!"

    # Generate summary paragraph with Gemini if available
    gemini_summary = ""
    if model:
        try:
            cars_info = "\n".join([
                f"- {r['make']} {r['model']} (₹{r['price']:,}, {r['user_rating']}/5 ⭐, {r['mileage']} kmpl)" 
                for r in recs_list[:3]
            ])
            prompt = f"""Write a friendly, encouraging paragraph (4-5 sentences) summarizing these car recommendations:
{cars_info}

Context: User budget is ₹{budget:,}, family size {family_size}, preference {preference}.
Make it conversational and explain why these are perfect fits. Be enthusiastic and helpful."""
            
            response = model.generate_content(prompt)
            gemini_summary = response.text.strip()
        except Exception as e:
            gemini_summary = f"We found {len(recs_list)} excellent cars that match your preferences perfectly! Each option has been carefully selected based on your budget, family size, and specific needs. Check out the details below and feel free to ask if you need more information about any model."

    return jsonify({
        "success": True,
        "filter_reason": reason,
        "recommendations": recs_list,
        "gemini_summary": gemini_summary,
        "count": len(recs_list),
        "user_message": user_message
    })


@app.route("/stats", methods=["GET"])
def stats():
    """Return dataset stats for the chatbot to show."""
    return jsonify({
        "total_cars": len(df),
        "price_range": {
            "min": int(df["price"].min()),
            "max": int(df["price"].max()),
            "avg": int(df["price"].mean())
        },
        "mileage_range": {
            "min": df["mileage"].min(),
            "max": df["mileage"].max(),
            "avg": round(df["mileage"].mean(), 1)
        },
        "makes": sorted(df["make"].unique().tolist()),
        "safety_ratings": sorted(df["safety_rating"].unique().tolist())
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

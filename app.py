"""
PantryChef AI
-------------
A Streamlit application that suggests a recipe based on ingredients you
already have at home, using a local LLM served via Ollama.

Required (same directory as this script):
    - ollama running locally, with a model already pulled (e.g. `ollama pull llama3.1`)

Optional:
    - sample_recipes.csv : a CSV with 'name' and 'ingredients' columns, used to
                            ground suggestions in real recipes via simple
                            ingredient-overlap retrieval.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import ollama
    
except ImportError:
    ollama = None

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATASET_PATH = Path("sample_recipes.csv")
DEFAULT_MODEL = "llama3.1:latest"
DEFAULT_STAPLES = ["salt", "pepper", "olive oil"]
NUM_GROUNDING_RECIPES = 3

st.set_page_config(
    page_title="PantryChef AI",
    page_icon="🍳",
    layout="centered",
)

st.write("Ollama model:", DEFAULT_MODEL)
# --------------------------------------------------------------------------- #
# Data loading (cached so files are read only once per session)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading recipe dataset...")
def load_dataset(path: Path) -> pd.DataFrame | None:
    """Load the optional recipe dataset used for grounding, if present."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "name" not in df.columns or "ingredients" not in df.columns:
        raise ValueError("Expected 'name' and 'ingredients' columns in the dataset.")
    return df


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #
def find_similar_recipes(
    ingredients: list[str], df: pd.DataFrame | None, n: int = NUM_GROUNDING_RECIPES
) -> list[tuple[str, str]]:
    """Return the top-n dataset recipes with the most ingredient overlap."""
    if df is None or not ingredients:
        return []

    tokens = {i.strip().lower() for i in ingredients if i.strip()}

    def score(ingredient_text: str) -> int:
        text = str(ingredient_text).lower()
        return sum(1 for t in tokens if t in text)

    scored = df.copy()
    scored["_score"] = scored["ingredients"].apply(score)
    scored = scored[scored["_score"] > 0].sort_values("_score", ascending=False).head(n)

    return [(str(row["name"]), str(row["ingredients"])) for _, row in scored.iterrows()]


def build_prompt(
    ingredients: list[str],
    staples: list[str],
    max_time: int,
    servings: int,
    grounding_recipes: list[tuple[str, str]],
) -> str:
    """Construct the recipe-generation prompt sent to the model."""
    all_ingredients = ", ".join(ingredients + staples)

    grounding_block = ""
    if grounding_recipes:
        examples = "\n".join(f'- "{name}": {ing}' for name, ing in grounding_recipes)
        grounding_block = f"\nSimilar real recipes for inspiration:\n{examples}\n"

    return f"""You are PantryChef, a home-cooking assistant.

Available ingredients: {all_ingredients}
{grounding_block}
Constraints:
- Servings: {servings}
- Maximum cooking time: {max_time} minutes

Suggest ONE recipe using as many available ingredients as possible. Respond in this format:

## [Recipe Name]
**Ingredients:**
- ...

**Steps:**
1. ...

**Estimated time:** [X minutes]
"""


def generate_recipe(prompt: str, model: str) -> str:
    """Send the prompt to Ollama and return the model's response text."""
    if ollama is None:
        raise RuntimeError("The 'ollama' package isn't installed. Run: pip install ollama")
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🍳 PantryChef AI")
    st.caption("List what's in your kitchen and get a recipe suggestion.")

    try:
        dataset = load_dataset(DATASET_PATH)
    except Exception as exc:  # noqa: BLE001 - surface any load error to the user
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    raw_ingredients = st.text_input(
        "Ingredients you have (comma-separated)",
        placeholder="chicken, rice, garlic, spinach",
    )
    col1, col2 = st.columns(2)
    with col1:
        max_time = st.slider("Max cooking time (minutes)", 10, 120, 30, step=5)
    with col2:
        servings = st.number_input("Servings", min_value=1, max_value=12, value=2)

    use_dataset = st.checkbox(
        "Ground suggestion in sample recipes",
        value=dataset is not None,
        disabled=dataset is None,
        help="Uses sample_recipes.csv, if present, to inspire the suggestion." if dataset is not None
        else "sample_recipes.csv not found in the app directory.",
    )

    if st.button("Suggest a Recipe", type="primary"):
        ingredients = [i.strip() for i in raw_ingredients.split(",") if i.strip()]

        if not ingredients:
            st.warning("Please enter at least one ingredient.")
            return

        grounding_recipes = find_similar_recipes(ingredients, dataset) if use_dataset else []
        prompt = build_prompt(ingredients, DEFAULT_STAPLES, max_time, servings, grounding_recipes)

        try:
            with st.spinner("PantryChef is thinking..."):
                recipe = generate_recipe(prompt, DEFAULT_MODEL)
        except Exception as exc:  # noqa: BLE001 - surface any generation error to the user
            st.error(f"Failed to generate a recipe: {exc}")
            return

        if grounding_recipes:
            st.caption("Inspired by: " + ", ".join(name for name, _ in grounding_recipes))

        st.markdown(recipe)


if __name__ == "__main__":
    main()

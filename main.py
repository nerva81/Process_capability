import streamlit as st
from pathlib import Path
import re

# Základní nastavení stránky
st.set_page_config(
    page_title="Quality Tools",
    page_icon="🧰",
    layout="wide",
)

st.title("🧰 Basic quality tools")
st.write("Chose tool in folder `pages/`.")

# -------------------------------------------------
# Funkce: hezčí název z názvu souboru
# -------------------------------------------------
def pretty_name_from_filename(fname: str) -> str:
    """
    Převede např. '1_Process_capability.py' -> 'Process capability'
    """
    stem = Path(fname).stem  # bez přípony
    # odstraň případné číselné prefixy typu '1_' '01_'
    stem = re.sub(r"^\d+[_-]*", "", stem)
    # nahradit podtržítka mezerami
    stem = stem.replace("_", " ").replace("-", " ")
    # první písmeno velké
    return stem.strip().capitalize()


# -------------------------------------------------
# Načtení stránek z adresáře pages
# -------------------------------------------------
BASE_DIR = Path(__file__).parent
PAGES_DIR = BASE_DIR / "pages"

if not PAGES_DIR.exists():
    st.error("Adresář `pages/` neexistuje. Ujisti se, že struktura projektu je správná.")
    st.stop()

page_files = sorted(PAGES_DIR.glob("*.py"))

if not page_files:
    st.warning("V adresáři `pages/` nejsou žádné *.py soubory.")
    st.stop()

# -------------------------------------------------
# Konfigurace dlaždic (volitelné – můžeš si doplnit popisy/ikony ručně)
# -------------------------------------------------
# Mapa: pattern ve jménu souboru -> (ikona, popis)
ICON_MAP = {
    "capab": ("📈", "Process capability evaluation (Cp, Cpk, Pp, Ppk)."),
    "measurement_system_analyze": ("📏", "Measurement system analyze (MSA) - Type1, Type2, Type3 and Attributive."),
    "pareto": ("📊", "Pareto analýza problémů."),
    "fishbone": ("🐟", "Ishikawa diagram příčin a následků."),
}

def guess_icon_and_desc(filepath: Path):
    name_lower = filepath.stem.lower()
    for key, (icon, desc) in ICON_MAP.items():
        if key in name_lower:
            return icon, desc
    # default
    return "🧩", "Nástroj kvality."

# -------------------------------------------------
# Vykreslení dlaždic – mřížka 3 sloupců
# -------------------------------------------------
NUM_COLS = 3
cols = st.columns(NUM_COLS)

for i, page_path in enumerate(page_files):
    col = cols[i % NUM_COLS]
    with col:
        icon, desc = guess_icon_and_desc(page_path)
        nice_name = pretty_name_from_filename(page_path.name)

        # "Obrázek" / head dlaždice
        tile = st.container(border=True)
        with tile:
            st.markdown(f"### {icon} {nice_name}")
            st.caption(desc)

            # Odkaz na stránku v multipage appce
            # cesta je relativní k main.py → "pages/xyz.py"
            st.page_link(
                f"pages/{page_path.name}",
                label="Open tool",
                icon="➡️",
                use_container_width=True,
            )

st.write("---")
st.caption(
    "Tip: Tools are in left sidebar. "
    "Mane page shows overview of the tools."
)


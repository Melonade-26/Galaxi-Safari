"""
Galaxy Classifier v3
====================
pip install -r requirements.txt
streamlit run streamlit_app.py
"""

import io
import os
import uuid
import random
import csv as csvmod
from datetime import datetime
from collections import defaultdict
from functools import lru_cache
import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from PIL import Image
import base64
# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

CSV_PATH     = "galaxy_values.csv"
IMAGE_DIR    = "pictures"
CATEGORY_DIR = "pictures_of_categories"
OUTPUT_PATH  = "ratings_output.csv"
USERS_PATH   = "users.csv"
LOGO_PATH    = "logo.png"        # nepovinné – ak súbor neexistuje, zobrazí sa len text

NAME         = "dr7objid"
RA_COL       = "ra_x"
DEC_COL      = "dec_x"
CLASS_COL    = "base_type"
MAIN_PARAMS  = ["GZ2_type", "class_from_conf", "confidence", "GZ1_type"]

CATEGORIES = [
    "Bars", "Bulge", "Dust_lane", "Dwarf_companions",
    "Extraplanar_features", "Flocculent_arms", "Grand_design_spiral_arms",
    "Jellyfish", "Nuclear_ring", "One_armed", "Ongoing_merger",
    "Polar_rings", "Ringed", "Superthin disk", "Tidal_features", "Warp",
]
SPECIAL_CATS = ["Bez štruktúry", "S", "Očistiť", "Neviem"]
ALL_CATS     = CATEGORIES + SPECIAL_CATS

CAT_DISPLAY = {
    "Bars":                     "Bars – priečne pásy",
    "Bulge":                    "Bulge – zhrubnutie",
    "Dust_lane":                "Dust lane – prašný pás",
    "Dwarf_companions":         "Dwarf companions",
    "Extraplanar_features":     "Extraplanar features",
    "Flocculent_arms":          "Flocculent arms",
    "Grand_design_spiral_arms": "Grand design spiral",
    "Jellyfish":                "Jellyfish",
    "Nuclear_ring":             "Nuclear ring",
    "One_armed":                "One-armed",
    "Ongoing_merger":           "Ongoing merger",
    "Polar_rings":              "Polar rings",
    "Ringed":                   "Ringed – prstencové",
    "Superthin disk":           "Superthin disk",
    "Tidal_features":           "Tidal features",
    "Warp":                     "Warp",
    "Bez štruktúry":            "Bez štruktúry",
    "S":                        "S – špirálovitá",
    "Očistiť":                  "Očistiť",
    "Neviem":                   "Neviem",
}

SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_IMG_PX    = 800

# ══════════════════════════════════════════════════════════════════
#  TRANSFORMÁCIE
# ══════════════════════════════════════════════════════════════════

def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - mn) / (mx - mn) * 255.0, 0, 255)

def _linear(ch):  return _norm(ch.astype(np.float64))
def _pow2(ch):    return _norm(np.power(ch.astype(np.float64), 2))
def _pow3(ch):    return _norm(np.power(ch.astype(np.float64), 3))
def _asinh(ch, a=0.1):
    x = ch.astype(np.float64)
    return _norm(np.arcsinh(x / a) / np.arcsinh(1.0 / a))

TRANSFORMS = {"Linear": _linear, "Pow²": _pow2, "Pow³": _pow3, "Asinh": _asinh}

def apply_transform(arr: np.ndarray, mode: str) -> np.ndarray:
    fn  = TRANSFORMS.get(mode, _linear)
    out = np.empty_like(arr, dtype=np.float64)
    for ch in range(min(3, arr.shape[2])):
        out[:, :, ch] = fn(arr[:, :, ch])
    return np.clip(out, 0, 255).astype(np.uint8)

# ══════════════════════════════════════════════════════════════════
#  OBRÁZKY
# ══════════════════════════════════════════════════════════════════

@lru_cache(maxsize=120)
def get_galaxy_img(objid: str, mode: str) -> bytes | None:
    """
    Vráti JPEG bytes galaxie s transformáciou.
    lru_cache = čistá RAM, O(1) lookup, zdieľané naprieč rerunmi.
    """
    path = os.path.join(IMAGE_DIR, f"galaxy_{objid}.jpg")
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    arr = apply_transform(np.array(img), mode)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=92)
    return buf.getvalue()

@lru_cache(maxsize=128)
def cat_first_img(cat: str) -> str | None:
    folder = os.path.join(CATEGORY_DIR, cat)
    if not os.path.isdir(folder):
        return None
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in SUPPORTED_IMG:
            return os.path.join(folder, f)
    return None

@lru_cache(maxsize=64)
def cat_all_imgs(cat: str) -> tuple:
    folder = os.path.join(CATEGORY_DIR, cat)
    if not os.path.isdir(folder):
        return ()
    return tuple(
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in SUPPORTED_IMG
    )

@lru_cache(maxsize=100)
def _img_b64(path: str) -> tuple[str, str]:
    """Načíta obrázok, zmenší na 240px, vráti (base64, mime). Kešované."""
    from PIL import Image as _PIL
    img = _PIL.open(path).convert("RGB")
    img.thumbnail((240, 240), _PIL.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80, optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

@lru_cache(maxsize=50)
def _cat_gallery_b64(cat: str, limit: int = 5) -> list[str]:
    """Vráti zoznam base64 JPEG reťazcov pre galériu kategórie (max limit obrázkov)."""
    result = []
    for path in cat_all_imgs(cat)[:limit]:
        try:
            b64, _ = _img_b64(path)
            result.append(f"data:image/jpeg;base64,{b64}")
        except Exception:
            pass
    return result


def build_injection_component(cats_img_data: dict) -> str:
    """
    HTML iframe komponent ktorý injektuje do Streamlit DOM:
    - ikonu náhľadu VPRAVO od textu checkboxu, vysokú ako samotný text (1em)
    - hover tooltip s väčším obrázkom
    - klik na ikonu otvorí modálnu galériu so všetkými obrázkami kategórie
    cats_img_data: {display_name: {"icon": "data:...", "gallery": ["data:...", ...]}}
    """
    import json
    data_json = json.dumps(cats_img_data, ensure_ascii=False)

    return f"""<!DOCTYPE html><html><body><script>
(function(){{
  const D = {data_json};
  const par = window.parent;

  // ── Tooltip ──────────────────────────────────────────────────────────
  function initUI() {{
    if (par.document.getElementById('_inj_tip')) return;

    // Tooltip
    const tip = par.document.createElement('div');
    tip.id = '_inj_tip';
    tip.style.cssText = 'display:none;position:fixed;z-index:99998;background:#1a1a2e;border:1px solid #555;border-radius:8px;padding:8px;box-shadow:0 6px 24px rgba(0,0,0,.7);pointer-events:none;';
    const tipImg = par.document.createElement('img');
    tipImg.id = '_inj_timg';
    tipImg.style.cssText = 'display:block;width:200px;height:200px;object-fit:contain;border-radius:4px;background:#000;';
    const tipLbl = par.document.createElement('div');
    tipLbl.id = '_inj_tlbl';
    tipLbl.style.cssText = 'color:#ccc;font-size:11px;text-align:center;margin-top:4px;';
    tip.appendChild(tipImg); tip.appendChild(tipLbl);
    par.document.body.appendChild(tip);

    // Modálna galéria
    const overlay = par.document.createElement('div');
    overlay.id = '_inj_overlay';
    overlay.style.cssText = 'display:none;position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,.75);';
    const box = par.document.createElement('div');
    box.id = '_inj_box';
    box.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:#1a1a2e;border:1px solid #555;border-radius:12px;padding:20px;max-width:700px;width:90vw;max-height:85vh;overflow-y:auto;';
    const hdr = par.document.createElement('div');
    hdr.style.cssText = 'display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;';
    const title = par.document.createElement('div');
    title.id = '_inj_gtitle';
    title.style.cssText = 'font-size:16px;font-weight:bold;color:#eee;';
    const closeBtn = par.document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = 'background:none;border:1px solid #666;color:#ccc;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:16px;';
    closeBtn.onclick = function(){{ overlay.style.display='none'; }};
    overlay.onclick = function(e){{ if(e.target===overlay) overlay.style.display='none'; }};
    const grid = par.document.createElement('div');
    grid.id = '_inj_grid';
    grid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;';
    hdr.appendChild(title); hdr.appendChild(closeBtn);
    box.appendChild(hdr); box.appendChild(grid);
    overlay.appendChild(box);
    par.document.body.appendChild(overlay);

    par.document.addEventListener('mousemove', function(e){{
      const t=par.document.getElementById('_inj_tip');
      if(t&&t.style.display!=='none'){{
        t.style.left=(e.clientX+18)+'px';
        t.style.top=Math.max(8,e.clientY-110)+'px';
      }}
    }});

  }}

  function openGallery(displayName, data) {{
    const overlay = par.document.getElementById('_inj_overlay');
    const title   = par.document.getElementById('_inj_gtitle');
    const grid    = par.document.getElementById('_inj_grid');
    if(!overlay||!title||!grid) return;
    title.textContent = displayName;
    grid.innerHTML = '';
    (data.gallery||[data.icon]).forEach(function(src){{
      const img = par.document.createElement('img');
      img.src = src;
      img.style.cssText = 'width:100%;aspect-ratio:1;object-fit:contain;background:#000;border-radius:6px;border:1px solid #444;';
      grid.appendChild(img);
    }});
    overlay.style.display = 'block';
  }}

  function inject() {{
    try {{
      initUI();
      const tip  = par.document.getElementById('_inj_tip');
      const timg = par.document.getElementById('_inj_timg');
      const tlbl = par.document.getElementById('_inj_tlbl');

      par.document.querySelectorAll('[data-testid="stCheckbox"]').forEach(function(cb){{
        if (cb.querySelector('img._inj')) return;
        const p = cb.querySelector('[data-testid="stWidgetLabel"] p')
               || cb.querySelector('label p')
               || cb.querySelector('label span:last-child');
        if (!p) return;
        const txt  = p.textContent.trim();
        const data = D[txt];
        if (!data) return;

        const im = par.document.createElement('img');
        im.src = data.icon;
        im.className = '_inj';
        im.title = txt;
        // Ešte väčšia ikona, pripnutá k pravému okraju CELÉHO riadku (label),
        // nie len k textu – takže je naozaj úplne vpravo v bloku kategórií.
        im.style.cssText = 'position:absolute;right:2px;top:50%;transform:translateY(-50%);height:2.6em;width:auto;border-radius:4px;background:#111;border:1px solid rgba(255,255,255,.25);cursor:pointer;transition:border-color .15s,transform .15s;z-index:5;';

        // Dôležité: na labeli meníme LEN position a padding – NIE display
        // ani width – to je presne to, čo predtým rozbilo riadok na dva.
        // position:relative len vytvorí ukotvenie pre absolútnu ikonu,
        // bez zásahu do interného flex rozloženia checkbox + text.
        const label = cb.querySelector('label') || p.parentElement;
        if (label) {{
          label.style.position     = 'relative';
          label.style.paddingRight = '3.4em';
        }}

        im.addEventListener('mouseenter', function(){{
          im.style.borderColor='#aaa'; im.style.transform='translateY(-50%) scale(1.2)';
          if(timg) timg.src=data.icon;
          if(tlbl) tlbl.textContent=txt;
          if(tip)  tip.style.display='block';
        }});
        im.addEventListener('mouseleave', function(){{
          im.style.borderColor='rgba(255,255,255,.25)'; im.style.transform='translateY(-50%)';
          if(tip) tip.style.display='none';
        }});
        im.addEventListener('click', function(e){{
          // preventDefault je nutný – klik kdekoľvek vnútri <label>
          // by inak (natívne správanie prehliadača) prepol aj checkbox.
          e.preventDefault();
          e.stopPropagation();
          if(tip) tip.style.display='none';
          openGallery(txt, data);
        }});
        im.addEventListener('mousedown', function(e){{ e.preventDefault(); }});

        p.appendChild(im);   // VPRAVO od textu
      }});
    }} catch(e) {{}}
  }}

  setTimeout(inject, 300);
  setTimeout(inject, 900);
  setTimeout(inject, 2500);
  new MutationObserver(inject).observe(
    par.document.body, {{childList:true, subtree:true}}
  );
}})();
</script></body></html>"""

# ══════════════════════════════════════════════════════════════════
#  DÁTA / CSV
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Načítavam katalóg galaxií…")
def load_df() -> pd.DataFrame:
    needed = set(MAIN_PARAMS) | {NAME, CLASS_COL, RA_COL, DEC_COL}
    avail  = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
    cols   = [c for c in avail if c in needed]
    df = pd.read_csv(CSV_PATH, usecols=cols, dtype={NAME: str}, engine="c")
    for col in [CLASS_COL] + MAIN_PARAMS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype("category")
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    return df

def load_results() -> dict:
    out = {}
    if not os.path.exists(OUTPUT_PATH):
        return out
    try:
        with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
            for row in csvmod.DictReader(f):
                k = (row.get(NAME, ""), row.get("user_id", ""))
                if k[0]:
                    out[k] = row
    except Exception:
        pass
    return out

def save_results(results: dict, df: pd.DataFrame):
    base   = [NAME] + [c for c in (RA_COL, DEC_COL) if c in df.columns]
    fields = base + ["user_id", "user_email", "user_name",
                     "questions", "note", "timestamp"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csvmod.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rec in results.values():
            w.writerow(rec)

def build_order(df: pd.DataFrame, results: dict, uid: str) -> list:
    """
    Rýchla vektorizovaná verzia – žiadne df.iloc v slučke.
    Triedy z base_type sa náhodne zamiešajú, v rámci každej triedy
    sa zachová poradie z CSV (= confidence desc).
    """
    rated = {k[0] for k in results if k[1] == uid}

    # Vyfiltruj neohodnotené riadky vektorovo
    mask     = ~df[NAME].isin(rated)
    filtered = df[mask]

    if CLASS_COL not in filtered.columns:
        idx = filtered.index.tolist()
        random.shuffle(idx)
        return idx

    # Zoskup podľa triedy – zachová poradie riadkov v CSV
    groups: dict[str, list] = {}
    for cls, grp in filtered.groupby(CLASS_COL, observed=True, sort=False):
        groups[str(cls)] = grp.index.tolist()

    keys = list(groups.keys())
    random.shuffle(keys)

    order: list[int] = []
    while any(groups[k] for k in keys):
        for k in keys:
            if groups[k]:
                order.append(groups[k].pop(0))
    return order

# ══════════════════════════════════════════════════════════════════
#  POUŽÍVATELIA
# ══════════════════════════════════════════════════════════════════

def get_or_create_user(email: str, name: str) -> dict:
    users: dict = {}
    if os.path.exists(USERS_PATH):
        try:
            with open(USERS_PATH, newline="", encoding="utf-8") as f:
                for row in csvmod.DictReader(f):
                    users[row["email"]] = row
        except Exception:
            pass
    if email in users:
        return users[email]
    user = {
        "user_id":     str(uuid.uuid5(uuid.NAMESPACE_URL, email)),
        "email":       email,
        "name":        name,
        "first_login": datetime.now().isoformat(timespec="seconds"),
    }
    new_file = not os.path.exists(USERS_PATH)
    with open(USERS_PATH, "a", newline="", encoding="utf-8") as f:
        w = csvmod.DictWriter(f, fieldnames=list(user.keys()))
        if new_file:
            w.writeheader()
        w.writerow(user)
    return user

# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════

def init_session(user: dict):
    uid = user["user_id"]
    if "df"        not in st.session_state: st.session_state.df        = load_df()
    if "results"   not in st.session_state: st.session_state.results   = load_results()
    if "transform" not in st.session_state: st.session_state.transform = "Linear"
    key_order = f"order_{uid}"
    key_pos   = f"pos_{uid}"
    if key_order not in st.session_state:
        st.session_state[key_order] = build_order(
            st.session_state.df, st.session_state.results, uid
        )
    if key_pos not in st.session_state:
        st.session_state[key_pos] = 0

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════

def render_sidebar(user: dict):
    uid   = user["user_id"]
    order = st.session_state.get(f"order_{uid}", [])
    pos   = st.session_state.get(f"pos_{uid}", 0)
    rated = sum(1 for k in st.session_state.results if k[1] == uid)
    total = len(order) + rated
    left  = max(0, len(order) - pos)

    transform_clicked = None   # zachytí klik na transform tlačidlo

    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown("## 🌌 Galaxy Classifier")

        st.divider()
        st.markdown(f"### 👤 {user['name']}")
        st.caption(f"ID: `{uid[:8]}…`")
        st.button("🚪 Odhlásiť sa", on_click=st.logout,
                  use_container_width=True, key="btn_logout")

        st.divider()
        st.markdown(f"**✅ Ohodnotených:** {rated}")
        st.markdown(f"**🔭 Zostatok:** {left}")
        if total > 0:
            st.progress(rated / total)

        st.divider()
        st.markdown("**🔬 Transformácia**")
        for mode in TRANSFORMS:
            active = st.session_state.transform == mode
            if st.button(
                f"{'▶ ' if active else ''}{mode}",
                key=f"tbtn_{mode}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                transform_clicked = mode

        st.divider()
        if os.path.exists(OUTPUT_PATH):
            # Cachujeme bytes v session_state – nečítame súbor pri každom rerun
            dl_key = "_dl_data"
            dl_size = os.path.getsize(OUTPUT_PATH)
            dl_cached_size = st.session_state.get("_dl_size", -1)
            if dl_key not in st.session_state or dl_size != dl_cached_size:
                with open(OUTPUT_PATH, "rb") as f:
                    st.session_state[dl_key] = f.read()
                st.session_state["_dl_size"] = dl_size
            st.download_button(
                "📥 Stiahnuť výsledky CSV",
                data=st.session_state[dl_key],
                file_name="ratings_output.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_download",
            )
        else:
            st.caption("(žiadne výsledky ešte)")

    # st.rerun() MIMO with st.sidebar: – vnútri kontextu spôsoboval zaseknutie
    if transform_clicked and transform_clicked != st.session_state.transform:
        st.session_state.transform = transform_clicked
        st.rerun()

# ══════════════════════════════════════════════════════════════════
#  HLAVNÁ ČASŤ
# ══════════════════════════════════════════════════════════════════

def render_main(user: dict):
    uid   = user["user_id"]
    df    = st.session_state.df
    order = st.session_state[f"order_{uid}"]
    pos   = st.session_state[f"pos_{uid}"]

    if not order or pos >= len(order):
        st.balloons()
        st.success("🎉 Všetky galaxie sú ohodnotené!")
        if os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, "rb") as f:
                st.download_button("📥 Stiahnuť výsledky", f.read(),
                                   "ratings_output.csv", "text/csv")
        return

    df_row    = df.iloc[order[pos]]
    objid     = str(df_row[NAME])
    prev      = st.session_state.results.get((objid, uid), {})
    prev_cats = set(q.strip() for q in prev.get("questions","").split(";") if q.strip())

    # Pre-zahrej cache raz pri prvom zobrazení
    if f"_pw_{objid}" not in st.session_state:
        st.session_state[f"_pw_{objid}"] = True
        for _m in TRANSFORMS:
            get_galaxy_img(objid, _m)

    # ── Navigácia NAD všetkým – top level, bezpečný rerun ─────────────────
    n1, n2, n3 = st.columns([1, 5, 1])
    go_prev = n1.button("⬅ Predošlá", use_container_width=True, key="nav_prev")
    n2.markdown(
        f"<div style='text-align:center;font-size:15px'>"
        f"<b>{objid}</b> &nbsp;·&nbsp; {pos+1} / {len(order)}</div>",
        unsafe_allow_html=True,
    )
    go_next = n3.button("Ďalšia ➡", use_container_width=True, key="nav_next")
    if go_prev and pos > 0:
        st.session_state[f"pos_{uid}"] -= 1
        st.rerun()
    if go_next and pos < len(order) - 1:
        st.session_state[f"pos_{uid}"] += 1
        st.rerun()

    # ── Dva stĺpce: obrázok | kategórie ──────────────────────────────────
    img_col, cat_col = st.columns([4, 2], gap="large")

    # ─── OBRÁZOK ─────────────────────────────────────────────────────────
    with img_col:
        img_bytes = get_galaxy_img(objid, st.session_state.transform)
        if img_bytes:
            st.image(img_bytes, use_container_width=True)
        else:
            st.error(f"Nenájdený: galaxy_{objid}.jpg")

        # Samostatný, kompaktný blok pod obrázkom s informáciami z CSV –
        # polia sú vedľa seba, blok nie je roztiahnutý na celú šírku obrázka.
        info_fields = [
            (CLASS_COL,   "base_type"),
            ("GZ1_type",  "gz1_type"),
            ("GZ2_type",  "gz2_type"),
        ]
        items = []
        for col_name, label in info_fields:
            if col_name in df_row.index:
                v = str(df_row[col_name])
                if v not in ("nan", "None", ""):
                    items.append((label, v))
        if items:
            cells = []
            for i, (label, v) in enumerate(items):
                border = (
                    "border-right:1px solid rgba(255,255,255,.15);"
                    if i < len(items) - 1 else ""
                )
                cells.append(
                    f"<div style='padding:0 16px;{border}text-align:center'>"
                    f"<div style='color:gray;font-size:11px;text-transform:uppercase;"
                    f"letter-spacing:.03em'>{label}</div>"
                    f"<div style='font-weight:600;font-size:14px;margin-top:2px'>{v}</div>"
                    f"</div>"
                )
            st.markdown(
                "<div style='text-align:center;margin-top:14px'>"
                "<div style='display:inline-flex;padding:8px 4px;"
                "background:rgba(255,255,255,.04);border-radius:8px;"
                "border:1px solid rgba(255,255,255,.1)'>"
                + "".join(cells) + "</div></div>",
                unsafe_allow_html=True,
            )

    # ─── KATEGÓRIE v st.form ─────────────────────────────────────────────
    cat_vals  = {}
    note_val  = ""
    submitted = False

    with cat_col:
        with st.form(key=f"form_{objid}_{uid}"):
            submitted = st.form_submit_button(
                "✅ Potvrdiť a ďalšia",
                type="primary",
                use_container_width=True,
            )
            st.markdown("**Kategórie**")
            for cat in CATEGORIES:
                cat_vals[cat] = st.checkbox(
                    CAT_DISPLAY.get(cat, cat),
                    value=(cat in prev_cats),
                )
            st.markdown("---")
            for cat in SPECIAL_CATS:
                cat_vals[cat] = st.checkbox(
                    CAT_DISPLAY.get(cat, cat),
                    value=(cat in prev_cats),
                )
            st.markdown("---")
            note_val = st.text_area(
                "Poznámka",
                value=prev.get("note", ""),
                height=70,
            )

    # ── Injekcia miniatúr cez iframe komponent (skripty tu naozaj bežia) ──
    inj_key = "_inj_html"
    if inj_key not in st.session_state:
        cats_img_data: dict[str, dict] = {}
        for cat in ALL_CATS:
            path = cat_first_img(cat)
            if path:
                try:
                    b64, mime = _img_b64(path)
                    icon_src  = f"data:{mime};base64,{b64}"
                    gallery   = _cat_gallery_b64(cat, limit=6)
                    display   = CAT_DISPLAY.get(cat, cat)
                    cats_img_data[display] = {
                        "icon":    icon_src,
                        "gallery": gallery if gallery else [icon_src],
                    }
                except Exception:
                    pass
        st.session_state[inj_key] = (
            build_injection_component(cats_img_data) if cats_img_data else ""
        )
    if st.session_state[inj_key]:
        components.html(st.session_state[inj_key], height=0)

    # Uloženie (mimo všetkých kontextov)
    if submitted:
        checked = {cat for cat, val in cat_vals.items() if val}
        _do_save(user, objid, df_row, checked, note_val)


def _do_save(user: dict, objid: str, row, checked: set, note: str):
    """Uloží hodnotenie a posunie na ďalšiu galaxiu. Volá sa mimo všetkých with blokov."""
    uid = user["user_id"]
    df  = st.session_state.df

    record = {
        NAME:         objid,
        "user_id":    uid,
        "user_email": user["email"],
        "user_name":  user["name"],
        "questions":  ";".join(sorted(checked)),
        "note":       note,
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
    }
    for c in (RA_COL, DEC_COL):
        if c in row.index:
            record[c] = row[c]

    st.session_state.results[(objid, uid)] = record
    # Invaliduj cache stiahnutia – nový obsah
    st.session_state.pop("_dl_data",  None)
    st.session_state.pop("_dl_size",  None)
    save_results(st.session_state.results, df)

    order = st.session_state[f"order_{uid}"]
    pos   = st.session_state[f"pos_{uid}"]
    if pos < len(order):
        del order[pos]
    if st.session_state[f"pos_{uid}"] >= len(order):
        st.session_state[f"pos_{uid}"] = max(0, len(order) - 1)

    st.rerun()

# ══════════════════════════════════════════════════════════════════
#  AUTH + ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def check_auth() -> dict:
    """Vráti user dict. Výsledok cachuje v session_state – disk sa číta raz."""
    if "_user" in st.session_state:
        return st.session_state["_user"]

    has_auth = False
    try:
        _ = st.secrets["auth"]
        has_auth = True
    except Exception:
        pass

    if not has_auth:
        user = get_or_create_user("dev@localhost", "Dev User")
    else:
        try:
            info = st.experimental_user
        except AttributeError:
            import importlib.metadata
            ver = importlib.metadata.version("streamlit")
            st.error(f"Streamlit {ver} nepodporuje Google OAuth. Potrebný ≥ 1.42.0")
            st.code("pip install --upgrade streamlit")
            st.stop()

        if not info.is_logged_in:
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                st.markdown("## 🌌 Galaxy Classifier")
                st.markdown("Prihlás sa svojím Google účtom.")
                st.button("🔑 Prihlásiť sa cez Google", type="primary",
                          on_click=st.login, kwargs={"provider": "google"},
                          use_container_width=True)
            st.stop()

        user = get_or_create_user(info.email, getattr(info, "name", info.email))

    st.session_state["_user"] = user
    return user


def main():
    st.set_page_config(
        page_title="Galaxy Classifier",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
<style>
/* Skryje Streamlit header lištu s Deploy tlačidlom. */
header[data-testid="stHeader"]       { display: none !important; }
[data-testid="stToolbar"]             { display: none !important; }
[data-testid="stDecoration"]          { display: none !important; }
[data-testid="stStatusWidget"]        { display: none !important; }
#MainMenu                             { visibility: hidden !important; }
footer                                { visibility: hidden !important; }

/* Sidebar je natrvalo viditeľný a nedá sa zbaliť – žiadne prepínacie
   tlačidlo (naše ani interné Streamlit) sa už s ním nedokázalo spoľahlivo
   pracovať naprieč verziami, preto je zbaľovanie úplne vypnuté. */
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    transform: none !important;
    min-width: 21rem !important;
    width: 21rem !important;
    margin-left: 0 !important;
}
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebar"] button[kind="header"] {
    display: none !important;
}

/* Väčšie okraje po stranách + obsah vycentrovaný, nech to nie je
   roztiahnuté cez celú šírku okna. */
.block-container {
    padding-top: 0.5rem !important;
    padding-left: 3.5rem !important;
    padding-right: 3.5rem !important;
    max-width: 1300px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
</style>""", unsafe_allow_html=True)

    user = check_auth()
    init_session(user)
    render_sidebar(user)
    render_main(user)


main()
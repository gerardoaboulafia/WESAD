import os, sys
import pygame
import pygame_menu
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from funciones import (
    extract_features_with_majority_label,
    data_cleaning,
    load_model,
    make_predictions,
    get_mongo_fs,
    list_patients,
    load_patient,
)

# ─── Constantes ───────────────────────────────────────────────
WIDTH, HEIGHT = 860, 640
GRAPH_FILE    = "prob_plot.png"
BANNER_PATH   = "/Users/gerardoaboulafia/Desktop/banner_wesad.png"

# ─── Colores y tema ──────────────────────────────────────────
BG        = (244, 239, 220)   
TITLE_BG  = (  0,  70, 140)   
FONT_COL  = ( 30,  30,  30)   
SELECTED  = (  0,  45, 110)   

custom_theme = pygame_menu.themes.THEME_DARK.copy()
custom_theme.title_background_color    = TITLE_BG
custom_theme.title_font_color          = FONT_COL
custom_theme.widget_font_color         = FONT_COL
custom_theme.selection_color           = SELECTED
custom_theme.widget_selection_effect.border_width = 0
custom_theme.widget_selection_effect.color        = SELECTED

try:
    custom_theme.background_color = pygame_menu.BaseImage(BANNER_PATH)
except pygame.error:
    custom_theme.background_color = BG

# ─── Pygame setup ─────────────────────────────────────────────
pygame.init()
surface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Predicción de Estrés – WESAD")

# ───────────────────────── MENÚ LOGIN ─────────────────────────
def make_login_menu():
    menu = pygame_menu.Menu("Login MongoDB", WIDTH, HEIGHT, theme=custom_theme)

    user_inp = menu.add.text_input("Usuario: ", default="")
    pwd_inp  = menu.add.text_input("Password: ", default="", password=True)
    msg_lbl  = menu.add.label("", font_size=20)

    def connect():
        user, pwd = user_inp.get_value(), pwd_inp.get_value()
        try:
            fs = get_mongo_fs(user, pwd)
            _ = fs.find_one()         # test rápido
            menu.clear()
            make_patient_menu(fs)
        except Exception as e:
            msg_lbl.set_title(f" {e}")
            msg_lbl.set_font(None, 20, (255, 80, 80))

    menu.add.button("Conectar", connect)
    menu.add.button("Salir", pygame_menu.events.EXIT)
    return menu

# ───────────────────────── MENÚ PACIENTES ─────────────────────
def make_patient_menu(fs):
    menu = pygame_menu.Menu("Seleccione paciente", WIDTH, HEIGHT, theme=custom_theme)
    pacientes = list_patients(fs)

    if not pacientes:
        menu.add.label("No hay archivos en GridFS")
    else:
        for fname in pacientes:
            menu.add.button(fname, lambda f=fname: run_pipeline(fs, f, menu))

    menu.add.button("Logout", make_login_menu)
    menu.mainloop(surface)

# ───────────────────── PIPELINE + GRÁFICO ─────────────────────
def run_pipeline(fs, filename, parent_menu):
    data_dict = load_patient(fs, filename)
    feats     = extract_features_with_majority_label(data_dict, verbose=False)
    feats     = data_cleaning(feats)
    model     = load_model()
    prob_df   = make_predictions(feats, model)

    # Guardar gráfico
    fig = plt.figure(figsize=(8, 3))
    plt.plot(prob_df["sample_idx"], prob_df["p_stress"])
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Muestra")
    plt.ylabel("P(estrés)")
    plt.title(f"{filename} – Probabilidad de estrés")
    plt.tight_layout()
    fig.savefig(GRAPH_FILE, dpi=100)
    plt.close(fig)

    show_result_screen(parent_menu, filename)

# ───────────────────── RESULT SCREEN ─────────────────────────
def show_result_screen(parent_menu, filename):
    menu = pygame_menu.Menu(filename, WIDTH, HEIGHT, theme=custom_theme)
    if os.path.exists(GRAPH_FILE):
        img = pygame_menu.BaseImage(GRAPH_FILE)
        menu.add.image(img)
    menu.add.button("⬅ Volver", parent_menu)
    menu.mainloop(surface)

# ───────────────────────── MAINLOOP ──────────────────────────
if __name__ == "__main__":
    make_login_menu().mainloop(surface)
    pygame.quit()
    sys.exit()

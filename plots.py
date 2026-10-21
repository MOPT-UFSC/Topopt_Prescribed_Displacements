import numpy as np
import matplotlib.pyplot as plt

def plot_history_from_txt():

    # ===================== leitura ======================
    outer_f0dyn, f0dyn = np.loadtxt("f0val_dyn.txt", unpack=True)
    outer_f0est, f0est = np.loadtxt("f0val_est.txt", unpack=True)
    outer_fval, fval = np.loadtxt("fval.txt", unpack=True)
    outer_bebeta, bebeta = np.loadtxt("bebeta.txt", unpack=True)
    outer_kkt, kktnorm = np.loadtxt("kktnorm.txt", unpack=True)
    outer_change, change = np.loadtxt("change.txt", unpack=True)

    # lenx = len(change)    
    # xmax = outer_change[lenx-1]

    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 12


    plt.figure()
    plt.plot(outer_f0dyn, f0dyn, label="LC1(a) - 380 Hz")
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel(r"(W[dB] / W[dB]) $\times$ 100", fontsize=18, labelpad=1)
    plt.title("Normalized active input power", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)

    plt.figure()
    plt.plot(outer_f0est, f0est, label="LC1(a) - 380 Hz")
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel(r"(J / J) $\times$ 100", fontsize=18, labelpad=1)
    plt.title("Normalized static compliance", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)

    plt.figure()
    plt.plot(outer_fval, fval, label="LC1(a) - 380 Hz")
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel(r"(m$^3$ / m$^3$) $\times$ 100", fontsize=18, labelpad=0)
    plt.title("Volume constraint", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)

    plt.figure()
    plt.plot(outer_bebeta, bebeta, label="LC1(a) - 380 Hz")
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel(r"$\beta$", fontsize=18, labelpad=1)
    plt.title(r"Projection parameter $\beta$", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)


    freq_i, f_init = np.loadtxt("FRF_init.txt", unpack=True)
    freq_o, f_opt = np.loadtxt("FRF_opt.txt", unpack=True)
    _, ux_init = np.loadtxt("Ux_init.txt", unpack=True)
    _, uy_init = np.loadtxt("Uy_init.txt", unpack=True)
    _, ux_opt = np.loadtxt("Ux_opt.txt", unpack=True)
    _, uy_opt = np.loadtxt("Uy_opt.txt", unpack=True)

    # plt.figure()
    # plt.semilogy(freq_i, f_init, label="Initial")
    # plt.semilogy(freq_o, f_opt, label="Optimized")
    # plt.grid(True)

    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Objective function")
    # plt.title("FRF comparison")
    # plt.legend()
    # plt.grid(True)

    plt.figure(figsize=(8, 6))
    plt.semilogy(freq_i, ux_init, label="Initial design - UX Tip", color='black', linewidth=1.2, alpha=0.8)
    plt.semilogy(freq_i, uy_init, label="Initial design - UX Tip", color='black', linewidth=1.2, alpha=0.8)
    plt.semilogy(freq_i, ux_opt, label="Initial design - UY Tip", color='green', linewidth=1.2, alpha=0.8)
    plt.semilogy(freq_i, uy_opt, label="Initial design - UY Tip", color='green', linewidth=1.2, alpha=0.8)

    #plt.semilogy(freq_i, uy_opt, label="Initial design - UY Tip", color='green', linewidth=1.2, alpha=0.8)

    # --- Definição das Faixas com Alturas Independentes ---
    # 'h_freq': altura dos textos verticais (Ex: 300 Hz)
    # 'h_letra': altura da letra da zona (Ex: A)
    bands = [
        {"range": [280, 380], "color": "blue", "label_text": "A", "h_freq": 0.95, "h_letra": 0.05},
        #{"range": [310, 390], "color": "blue", "label_text": "A", "h_freq": 0.95, "h_letra": 0.05},
        {"range": [1050, 1180], "color": "blue", "label_text": "B", "h_freq": 0.95, "h_letra": 0.05},
        #{"range": [1120, 1200], "color": "blue", "label_text": "B", "h_freq": 0.95, "h_letra": 0.05},
        #{"range": [1320, 1440], "color": "red", "label_text": "C", "h_freq": 0.45, "h_letra": 0.05}
        #{"range": [1360, 1440], "color": "red", "label_text": "C", "h_freq": 0.40, "h_letra": 0.05}
    ]

    offset = 15
    for b in bands:
        x1, x2 = b["range"]
        
        # 1. Área e Linhas
        plt.axvspan(x1, x2, color=b["color"], alpha=0.15)
        plt.axvline(x=x1, color=b["color"], linestyle='--', alpha=0.5)
        plt.axvline(x=x2, color=b["color"], linestyle='--', alpha=0.5)
        
        # 2. Texto do limite INFERIOR (Usa a altura 'h_freq' de cada banda)
        plt.text(x1 - offset, b["h_freq"], f'{x1} Hz', color=b["color"], 
                transform=plt.gca().get_xaxis_transform(),
                rotation=90, va='top', ha='right', fontsize=14, fontweight='bold')

        # 3. Texto do limite SUPERIOR (Usa a altura 'h_freq' de cada banda)
        plt.text(x2 + offset, b["h_freq"], f'{x2} Hz', color=b["color"], 
                transform=plt.gca().get_xaxis_transform(),
                rotation=90, va='top', ha='left', fontsize=14, fontweight='bold')

        # 4. Letras A, B e C (Usa a altura 'h_letra' de cada banda)
        x_center = (x1 + x2) / 2
        plt.text(x_center, b["h_letra"], b["label_text"], color=b["color"],
                transform=plt.gca().get_xaxis_transform(),
                fontsize=14, fontweight='bold', ha='center', va='bottom')

    # --- Formatação Final ---
    plt.xlabel("Frequency [Hz]", fontsize=18)
    plt.ylabel("Tip displacement [m]", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)



    plt.tight_layout()
    plt.show()

#plot_history_from_txt()

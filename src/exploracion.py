import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#encuestas, respiban, outliers

def mean_std_format(series):
    return f"{series.mean():.1f}±{series.std():.1f}"

def plot_chunks(ecg_signal, labels, signal_name, paciente, chunk_size=700, colormap='tab10'):
    # Aseguramos que la señal sea 1D
    if ecg_signal.ndim == 2:
        ecg_signal = ecg_signal.ravel()

    n_samples = len(ecg_signal)
    n_chunks  = n_samples // chunk_size
    labels    = labels[:n_samples]

    # 1) Calculamos la etiqueta representativa de cada bloque
    segment_labels = []
    for i in range(n_chunks):
        start = i * chunk_size
        end   = start + chunk_size
        seg_lbl = int(np.round(np.median(labels[start:end])))
        segment_labels.append(seg_lbl)

    # 2) Asignar colores: labels 1 y 2 se colorean con el colormap, el resto en gris
    cmap = plt.get_cmap(colormap)
    label_to_color = {
        1: cmap(0),
        2: cmap(1)
    }
    default_color = 'gray'

    # 3) Dibujamos cada bloque con su color según la etiqueta
    plt.figure(figsize=(22, 8))
    for i, seg_lbl in enumerate(segment_labels):
        start = i * chunk_size
        end   = start + chunk_size
        x     = np.arange(start, end)
        color = label_to_color.get(seg_lbl, default_color)
        y     = ecg_signal[start:end]
        plt.plot(x, y, color=color, linewidth=0.8)

    # 4) Construir leyenda: entradas separadas para 1 y 2 y una entrada combinada para "Otros"
    legend_elems = []
    if 1 in segment_labels:
        legend_elems.append(Line2D([0], [0], color=label_to_color[1], lw=3, label='Baseline'))
    if 2 in segment_labels:
        legend_elems.append(Line2D([0], [0], color=label_to_color[2], lw=3, label='Estrés'))
    # Verificar si existen etiquetas distintas de 1 y 2
    otros = [lbl for lbl in set(segment_labels) if lbl not in (1, 2)]
    if otros:
        legend_elems.append(Line2D([0], [0], color=default_color, lw=3, label='Otros'))

    plt.legend(handles=legend_elems, title="Labels", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.title(f'{signal_name} por bloques de {chunk_size} muestras. Paciente {paciente}')
    plt.xlabel("Muestra")
    plt.ylabel("Amplitud señal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()    # Asegurar que la señal sea 1D
    if ecg_signal.ndim == 2:
        ecg_signal = ecg_signal.ravel()

    n_samples = len(ecg_signal)
    n_chunks  = n_samples // chunk_size
    labels    = labels[:n_samples]

    # Calcular la etiqueta representativa de cada bloque
    segment_labels = []
    for i in range(n_chunks):
        start = i * chunk_size
        end   = start + chunk_size
        seg_lbl = int(np.round(np.median(labels[start:end])))
        segment_labels.append(seg_lbl)

    # Asignar colores: labels 1 y 2 se colorean con el colormap, el resto en gris
    cmap = plt.get_cmap(colormap)
    label_to_color = {
        1: cmap(0),
        2: cmap(1)
    }
    default_color = 'gray'

    # Dibujar cada bloque con su color según la etiqueta
    plt.figure(figsize=(22, 8))
    for i, seg_lbl in enumerate(segment_labels):
        start = i * chunk_size
        end   = start + chunk_size
        x     = np.arange(start, end)
        color = label_to_color.get(seg_lbl, default_color)
        y     = ecg_signal[start:end]
        plt.plot(x, y, color=color, linewidth=0.8)

    # Construir leyenda: entradas separadas para 1 y 2 y una entrada combinada para "Otros"
    legend_elems = []
    if 1 in segment_labels:
        legend_elems.append(Line2D([0], [0], color=label_to_color[1], lw=3, label='Baseline'))
    if 2 in segment_labels:
        legend_elems.append(Line2D([0], [0], color=label_to_color[2], lw=3, label='Estrés'))
    # Verificar si existen etiquetas distintas de 1 y 2
    otros = [lbl for lbl in set(segment_labels) if lbl not in (1, 2)]
    if otros:
        legend_elems.append(Line2D([0], [0], color=default_color, lw=3, label='Otros'))

    plt.legend(handles=legend_elems, title="Labels", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.title(f'{signal_name} por bloques de {chunk_size} muestras. Paciente {paciente}')
    plt.xlabel("Muestra")
    plt.ylabel("Amplitud señal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_multisignal_chunks(signals_dict, labels, paciente=None, chunk_size=700, colormap='tab10', figsize_base=4):

    labels = np.asarray(labels)
    cmap   = plt.get_cmap(colormap)
    label_to_color = {1: cmap(0), 2: cmap(1)}
    default_color  = 'gray'

    for signal_name, sig in signals_dict.items():
        sig = np.asarray(sig)
        if sig.ndim == 1:
            sig = sig[:, None]               # → (n, 1)

        n_samples, n_channels = sig.shape
        usable_len = min(n_samples, len(labels))
        n_chunks   = usable_len // chunk_size

        # Etiqueta dominante de cada bloque
        seg_labels = [
            int(np.round(np.median(labels[i*chunk_size:(i+1)*chunk_size])))
            for i in range(n_chunks)
        ]

        # Figura y ejes
        fig, axes = plt.subplots(
            n_channels, 1, sharex=True,
            figsize=(22, figsize_base * n_channels)
        )
        if n_channels == 1:
            axes = [axes]

        # Trazar cada canal
        for ch, ax in enumerate(axes):
            for blk_idx, lbl in enumerate(seg_labels):
                st, en = blk_idx * chunk_size, (blk_idx + 1) * chunk_size
                x = np.arange(st, en)
                y = sig[st:en, ch]
                ax.plot(x, y,
                        color=label_to_color.get(lbl, default_color),
                        linewidth=0.8)
            ax.set_title(f'{signal_name} – Canal {ch} - Paciente {paciente}')
            ax.set_ylabel(f'Amplitud {signal_name}')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Muestra')

        # Leyenda de los gráficos
        legend_elems = []
        if 1 in seg_labels:
            legend_elems.append(Line2D([0], [0], color=label_to_color[1],
                                       lw=3, label='Baseline'))
        if 2 in seg_labels:
            legend_elems.append(Line2D([0], [0], color=label_to_color[2],
                                       lw=3, label='Estrés'))
        if any(lbl not in (1, 2) for lbl in seg_labels):
            legend_elems.append(Line2D([0], [0], color=default_color,
                                       lw=3, label='Otros'))

        fig.legend(handles=legend_elems, title='Labels',
                   bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.show()

    labels = np.asarray(labels)
    cmap   = plt.get_cmap(colormap)
    label_to_color = {1: cmap(0), 2: cmap(1)}
    default_color  = 'gray'

    for signal_name, sig in signals_dict.items():
        sig = np.asarray(sig)
        if sig.ndim == 1:
            sig = sig[:, None]               # → (n, 1)

        n_samples, n_channels = sig.shape
        usable_len = min(n_samples, len(labels))
        n_chunks   = usable_len // chunk_size

        # Etiqueta dominante de cada bloque
        seg_labels = [
            int(np.round(np.median(labels[i*chunk_size:(i+1)*chunk_size])))
            for i in range(n_chunks)
        ]

        # Figura y ejes
        fig, axes = plt.subplots(
            n_channels, 1, sharex=True,
            figsize=(22, figsize_base * n_channels)
        )
        if n_channels == 1:
            axes = [axes]

        # Trazar cada canal
        for ch, ax in enumerate(axes):
            for blk_idx, lbl in enumerate(seg_labels):
                st, en = blk_idx * chunk_size, (blk_idx + 1) * chunk_size
                x = np.arange(st, en)
                y = sig[st:en, ch]
                ax.plot(x, y,
                        color=label_to_color.get(lbl, default_color),
                        linewidth=0.8)
            ax.set_title(f'{signal_name} – Canal {ch} - Paciente {paciente}')
            ax.set_ylabel(f'Amplitud {signal_name}')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Muestra')

        # Leyenda de los gráficos
        legend_elems = []
        if 1 in seg_labels:
            legend_elems.append(Line2D([0], [0], color=label_to_color[1],
                                       lw=3, label='Baseline'))
        if 2 in seg_labels:
            legend_elems.append(Line2D([0], [0], color=label_to_color[2],
                                       lw=3, label='Estrés'))
        if any(lbl not in (1, 2) for lbl in seg_labels):
            legend_elems.append(Line2D([0], [0], color=default_color,
                                       lw=3, label='Otros'))

        fig.legend(handles=legend_elems, title='Labels',
                   bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.show()

def extract_float(val):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return float(val[0])
    if isinstance(val, str):
        # Sacar corchetes y convertir a float
        val = val.strip("[]")
        return float(val)
    return float(val)

def porcentaje_outliers(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    porcentaje = len(outliers) / len(df) * 100
    return porcentaje
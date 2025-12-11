# Analyse: SPL-Daten-Verfälschung in Plot3DSPL_new.py

## 🔴 Identifizierte Probleme

### Problem 1: Clipping VOR der Quantisierung (Zeile 1129)

**Code:**
```python
# Color step: Clipping + Quantisierung
if is_step_mode:
    # Clipping vor der Quantisierung, um Ausreißer außerhalb der Colorbar zu verhindern
    spl_plot = np.clip(spl_plot, cbar_min, cbar_max)  # ❌ VERFÄLSCHT DIE DATEN!
    scalars = self._quantize_to_steps(spl_plot, cbar_step)
```

**Problem:**
- Die originalen SPL-Werte werden auf `cbar_min`/`cbar_max` begrenzt
- Wenn originale Werte außerhalb der Colorbar liegen (z.B. 130 dB bei cbar_max=120 dB), werden sie auf 120 dB gesetzt
- **Das verfälscht die Daten**, bevor sie quantisiert werden
- Die Quantisierung arbeitet dann mit bereits verfälschten Werten

**Vergleich mit Plot3DSPL.py:**
```python
# Zeile 1020: Gibt mehr Spielraum
plot_values = np.clip(plot_values, cbar_min - 20, cbar_max + 20)
```

---

### Problem 2: Clipping bei Triangulation (Zeile 1234)

**Code:**
```python
if is_step_mode:
    # Verwende originale Grid-Koordinaten (nicht upgescalte)
    points_orig = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    # 🎯 WICHTIG: Begrenze Werte auf cbar_min/cbar_max VOR Quantisierung!
    spl_values_clipped = np.clip(spl_values.ravel(), cbar_min, cbar_max)  # ❌ VERFÄLSCHT!
    # Quantisiere die begrenzten SPL-Werte VOR der Interpolation
    spl_orig_quantized = self._quantize_to_steps(spl_values_clipped, cbar_step)
```

**Problem:**
- Die **originalen Grid-Werte** (`spl_values`) werden geclippt
- Das verfälscht die Daten **bevor** sie auf die Vertex-Positionen interpoliert werden
- Die Interpolation arbeitet dann mit bereits verfälschten Werten

---

### Problem 3: Redundantes Clipping nach Triangulation (Zeile 1341)

**Code:**
```python
combined_scalars = np.concatenate(all_scalars)
# Clipping auf Colorbar-Bereich vor Übergabe an PyVista
combined_scalars = np.clip(combined_scalars, cbar_min, cbar_max)  # ❌ REDUNDANT!
```

**Problem:**
- Die Werte wurden bereits vorher geclippt und quantisiert
- Dieses Clipping ist redundant und könnte bereits quantisierte Werte verfälschen
- Wenn Werte bereits quantisiert sind, sollten sie bereits im richtigen Bereich sein

---

### Problem 4: Clipping in time_mode und phase_mode (Zeile 1005, 1010)

**Code:**
```python
if time_mode:
    spl_values = np.real(sound_field_p_complex)
    spl_values = np.nan_to_num(spl_values, nan=0.0, posinf=0.0, neginf=0.0)
    spl_values = np.clip(spl_values, cbar_min, cbar_max)  # ❌ VERFÄLSCHT!
elif phase_mode:
    spl_values = np.angle(sound_field_p_complex)
    spl_values = np.nan_to_num(spl_values, nan=0.0, posinf=0.0, neginf=0.0)
    spl_values = np.clip(spl_values, cbar_min, cbar_max)  # ❌ VERFÄLSCHT!
```

**Problem:**
- Die originalen Werte werden sofort geclippt
- Das verfälscht die Daten **bevor** sie verarbeitet werden
- Spätere Interpolationen arbeiten mit bereits verfälschten Werten

---

## ✅ Vergleich mit Plot3DSPL.py

**Plot3DSPL.py (Zeile 1014-1020):**
```python
# Clipping nur für Visualisierung (nicht für Sampling)
if time_mode:
    plot_values = np.clip(plot_values, cbar_min, cbar_max)
elif phase_mode:
    plot_values = np.clip(plot_values, cbar_min, cbar_max)
else:
    plot_values = np.clip(plot_values, cbar_min - 20, cbar_max + 20)  # ✅ Mehr Spielraum!
```

**Unterschiede:**
- Plot3DSPL.py gibt **20 dB Spielraum** für SPL-Modus
- Clipping erfolgt **nach** der Verarbeitung, nicht vorher
- Originale Werte bleiben länger erhalten

---

## 🎯 Empfohlene Lösungen

### Lösung 1: Clipping nur für Visualisierung (nicht für Verarbeitung)

**Vorher (Zeile 1129):**
```python
# ❌ FALSCH: Clipping vor Quantisierung
spl_plot = np.clip(spl_plot, cbar_min, cbar_max)
scalars = self._quantize_to_steps(spl_plot, cbar_step)
```

**Nachher:**
```python
# ✅ RICHTIG: Quantisierung ohne Clipping
scalars = self._quantize_to_steps(spl_plot, cbar_step)
# Clipping nur für Visualisierung (PyVista)
scalars = np.clip(scalars, cbar_min, cbar_max)
```

---

### Lösung 2: Kein Clipping bei Triangulation

**Vorher (Zeile 1234):**
```python
# ❌ FALSCH: Clipping vor Quantisierung
spl_values_clipped = np.clip(spl_values.ravel(), cbar_min, cbar_max)
spl_orig_quantized = self._quantize_to_steps(spl_values_clipped, cbar_step)
```

**Nachher:**
```python
# ✅ RICHTIG: Quantisierung ohne Clipping
spl_orig_quantized = self._quantize_to_steps(spl_values.ravel(), cbar_step)
# Clipping nur für Visualisierung
spl_orig_quantized = np.clip(spl_orig_quantized, cbar_min, cbar_max)
```

---

### Lösung 3: Mehr Spielraum wie in Plot3DSPL.py

**Vorher:**
```python
# ❌ FALSCH: Strikte Begrenzung
spl_plot = np.clip(spl_plot, cbar_min, cbar_max)
```

**Nachher:**
```python
# ✅ RICHTIG: Mehr Spielraum (wie Plot3DSPL.py)
spl_plot = np.clip(spl_plot, cbar_min - 20, cbar_max + 20)
```

---

### Lösung 4: Clipping nur am Ende (vor PyVista)

**Empfehlung:**
- **Kein Clipping** während der Verarbeitung
- **Quantisierung** ohne Clipping
- **Clipping nur am Ende** vor Übergabe an PyVista (für Visualisierung)

**Code-Struktur:**
```python
# 1. Originale Werte laden (KEIN Clipping)
spl_values = self.functions.mag2db(pressure_magnitude)

# 2. Optional: Upscaling (KEIN Clipping)
if PLOT_UPSCALE_FACTOR > 1:
    spl_fine = self._bilinear_interpolate_grid(...)
    spl_plot = spl_fine
else:
    spl_plot = spl_values

# 3. Quantisierung (KEIN Clipping vorher)
if is_step_mode:
    scalars = self._quantize_to_steps(spl_plot, cbar_step)
else:
    scalars = spl_plot

# 4. Clipping nur für Visualisierung (am Ende)
scalars = np.clip(scalars, cbar_min, cbar_max)
```

---

## 📊 Zusammenfassung

**Hauptprobleme:**
1. ❌ Clipping **vor** Quantisierung verfälscht die Daten
2. ❌ Clipping bei Triangulation verfälscht originale Grid-Werte
3. ❌ Redundantes Clipping nach Quantisierung
4. ❌ Clipping in time_mode/phase_mode verfälscht Daten sofort

**Empfehlung:**
- ✅ Clipping **nur am Ende** für Visualisierung
- ✅ Quantisierung **ohne** vorheriges Clipping
- ✅ Mehr Spielraum wie in Plot3DSPL.py (cbar_min - 20, cbar_max + 20)
- ✅ Originale Werte so lange wie möglich erhalten

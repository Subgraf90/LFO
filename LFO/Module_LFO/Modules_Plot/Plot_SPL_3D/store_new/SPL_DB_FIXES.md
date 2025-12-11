# Fixes: dB-Werte korrekt plotten mit ColorbarManager

## ✅ Implementierte Fixes

### Fix 1: Clipping in time_mode/phase_mode entfernt (Zeile 1008, 1013)

**Vorher:**
```python
spl_values = np.clip(spl_values, cbar_min, cbar_max)  # ❌ Verfälscht sofort
```

**Nachher:**
```python
# Clipping nur für Visualisierung am Ende, nicht hier
# Originale Werte bleiben erhalten
```

**Zusammenhang mit ColorbarManager:**
- ColorbarManager verwendet `cbar_min`, `cbar_max` aus `get_colorbar_params()`
- Originale Werte müssen erhalten bleiben, damit ColorbarManager korrekt arbeitet

---

### Fix 2: Clipping vor Quantisierung entfernt (Zeile 1132)

**Vorher:**
```python
spl_plot = np.clip(spl_plot, cbar_min, cbar_max)  # ❌ Verfälscht vor Quantisierung
scalars = self._quantize_to_steps(spl_plot, cbar_step)
```

**Nachher:**
```python
# 🎯 FIX: Quantisierung OHNE vorheriges Clipping - Originale dB-Werte erhalten
# ColorbarManager verwendet cbar_min/cbar_max/cbar_step - Werte müssen nicht vorher geclippt werden
scalars = self._quantize_to_steps(spl_plot, cbar_step)
```

**Zusammenhang mit ColorbarManager:**
- ColorbarManager erstellt Levels: `np.arange(cbar_min, cbar_max + cbar_step * 0.5, cbar_step)`
- Quantisierung muss mit originalen Werten arbeiten, damit Levels korrekt sind
- `_quantize_to_steps()` arbeitet mit `cbar_step` aus ColorbarManager

---

### Fix 3: Clipping bei Triangulation entfernt (Zeile 1237)

**Vorher:**
```python
spl_values_clipped = np.clip(spl_values.ravel(), cbar_min, cbar_max)  # ❌ Verfälscht originale Grid-Werte
spl_orig_quantized = self._quantize_to_steps(spl_values_clipped, cbar_step)
```

**Nachher:**
```python
# 🎯 FIX: Quantisierung OHNE vorheriges Clipping - Originale dB-Werte erhalten
# ColorbarManager verwendet cbar_min/cbar_max/cbar_step - Werte müssen nicht vorher geclippt werden
spl_orig_quantized = self._quantize_to_steps(spl_values.ravel(), cbar_step)
```

**Zusammenhang mit ColorbarManager:**
- Originale Grid-Werte müssen erhalten bleiben
- Quantisierung arbeitet mit `cbar_step` aus ColorbarManager

---

### Fix 4: Clipping bei vertikalen Surfaces entfernt (Zeile 1932, 1936, 2485)

**Vorher:**
```python
spl_values = np.clip(spl_values, cbar_min, cbar_max)  # ❌ Verfälscht vor Quantisierung
spl_plot = np.clip(spl_plot, cbar_min, cbar_max)  # ❌ Verfälscht vor Quantisierung
```

**Nachher:**
```python
# Clipping nur für Visualisierung am Ende, nicht hier
# 🎯 FIX: Quantisierung OHNE vorheriges Clipping - Originale dB-Werte erhalten
scalars = self._quantize_to_steps(spl_plot, cbar_step)
```

**Zusammenhang mit ColorbarManager:**
- Vertikale Surfaces verwenden die gleichen Colorbar-Parameter
- Konsistenz mit horizontalen Surfaces

---

### Fix 5: Kommentare für verbleibende Clipping-Stellen (nur Visualisierung)

**Angepasste Stellen:**
- Zeile 1348: `combined_scalars` (nach Triangulation)
- Zeile 1404, 1431: `scalars_for_mesh` (Fallback build_surface_mesh)
- Zeile 2511: `scalars_for_mesh` (vertikale Surfaces)

**Kommentare hinzugefügt:**
```python
# 🎯 FIX: Clipping nur für Visualisierung (vor build_surface_mesh/PyVista)
# ColorbarManager verwendet cbar_min/cbar_max - Clipping hier ist OK für Darstellung
```

---

## 📊 Zusammenhang mit ColorbarManager

### ColorbarManager.get_colorbar_params()

```python
def get_colorbar_params(self, phase_mode: bool) -> dict[str, float]:
    rng = self.settings.colorbar_range
    return {
        'min': float(rng['min']),      # z.B. 40.0 dB
        'max': float(rng['max']),      # z.B. 120.0 dB
        'step': float(rng['step']),    # z.B. 5.0 dB
        'tick_step': float(rng['tick_step']),
        'label': "SPL (dB)",
    }
```

### ColorbarManager.render_colorbar()

**Color Step Modus:**
```python
# Erstellt Levels: min, min+step, min+2*step, ..., max
levels = np.arange(cbar_min, cbar_max + cbar_step * 0.5, cbar_step)
# Immer genau 11 Levels (10 Farben)
```

**Wichtig:**
- Levels müssen auf Vielfachen von `cbar_step` ausgerichtet sein
- `max = min + (step * 10)` für 11 Levels
- Quantisierung muss mit originalen Werten arbeiten, damit Levels korrekt sind

---

## 🎯 Ergebnis

**Vorher:**
- ❌ dB-Werte wurden mehrfach geclippt
- ❌ Originale Werte wurden verfälscht
- ❌ Quantisierung arbeitete mit bereits verfälschten Werten
- ❌ ColorbarManager Levels stimmten nicht mit geplotteten Werten überein

**Nachher:**
- ✅ Originale dB-Werte bleiben so lange wie möglich erhalten
- ✅ Quantisierung arbeitet mit originalen Werten
- ✅ ColorbarManager Levels stimmen mit geplotteten Werten überein
- ✅ Clipping nur am Ende für Visualisierung (PyVista)
- ✅ Konsistenz zwischen ColorbarManager und Plot-Daten

---

## 🔍 Validierung

**Prüfe:**
1. Originale dB-Werte werden korrekt aus `sound_field_p_complex` konvertiert
2. Quantisierung verwendet `cbar_step` aus ColorbarManager
3. Clipping nur am Ende für Visualisierung
4. ColorbarManager Levels stimmen mit geplotteten Werten überein

**Debug-Ausgaben:**
- Zeigt originale SPL-Werte vor Quantisierung
- Zeigt Colorbar Range: `[cbar_min, cbar_max] dB, Step: cbar_step dB`
- Zeigt quantisierte Werte nach Quantisierung

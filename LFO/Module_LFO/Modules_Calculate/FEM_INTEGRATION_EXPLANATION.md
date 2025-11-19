# Integration Points auf Facets - Erklärung

## Problem: Aktuelle Approximation

**Was wir aktuell machen:**
```python
# Vereinfachte Methode:
area_per_dof = panel_area / float(panel_dof_indices.size)
dof_value = neumann_value * area_per_dof
b.array[panel_dof_indices] += dof_value
```

**Problem:**
- Wir verteilen die Panel-Fläche gleichmäßig auf die DOFs
- Wir nehmen an: `φ_i(x_j) = 1` wenn `i == j`, sonst `0`
- **Das ist nur eine Approximation!**

## Was ist eine exakte Integration?

### Mathematisch:

Für das Flächenintegral: `∫_S (∂p/∂n) * φ_i ds`

Die **korrekte numerische Integration** verwendet:
```
∫_S f(x) ds ≈ Σ_j f(x_j) * w_j * |J_j|
```

wobei:
- `x_j` = Integration Points (Stützstellen auf der Facet-Oberfläche)
- `w_j` = Gewichtungen (Quadratur-Gewichte)
- `|J_j|` = Jakobische Determinante (Flächenelement)

### Beispiel für ein 2D-Dreieck (Facet in 3D):

```
    3
    |\
    | \
    |  \
    1---2
```

**Integration Points** könnten sein:
- Mittelpunkt des Dreiecks
- Gauß-Punkte (z.B. 3 Punkte für quadratische Integration)

**Testfunktion** `φ_i(x)` hat unterschiedliche Werte:
- `φ_1` an Punkt 1: `φ_1(x_1) = 1.0`
- `φ_1` an Punkt 2: `φ_1(x_2) = 0.0`
- `φ_1` am Mittelpunkt: `φ_1(x_mid) = 0.33...` (bei P1-Elementen)

## Was bedeutet das in dolfinx/FEniCS?

### Aktuell (Approximation):

```python
# Wir kennen nur die DOF-Koordinaten
panel_dof_indices = [123, 124, 125, ...]  # Indizes der DOFs
area_per_dof = panel_area / len(panel_dof_indices)

# Vereinfacht: b_i += (∂p/∂n) * (area / num_dofs)
for dof_idx in panel_dof_indices:
    b.array[dof_idx] += neumann_value * area_per_dof
```

**Problem:** Wir ignorieren, dass:
- Testfunktionen `φ_i` unterschiedliche Werte an verschiedenen Punkten haben
- Die tatsächliche Facet-Geometrie könnte unregelmäßig sein
- Die Fläche pro DOF ist nicht gleich der Integration-Gewichtung

### Korrekt (mit Integration Points):

```python
# 1. Finde Facets, die zur Panel-Fläche gehören
#    (müssen als Facet-Tags im Mesh markiert sein)
panel_facets = mesh.facets_with_tag(panel_tag)

# 2. Für jedes Facet: Finde Integration Points
from dolfinx import fem
from dolfinx.fem import IntegralType

# Integration Points für Facet-Integration
quadrature_degree = 2  # Grad der Quadratur-Formel
ip_coords, ip_weights = get_facet_integration_points(mesh, panel_facets, quadrature_degree)

# 3. Für jeden Integration Point:
for facet_idx in panel_facets:
    # Hole Integration Points und Gewichtungen für dieses Facet
    for ip in integration_points[facet_idx]:
        x_ip = ip.coordinate  # Koordinate des Integration Points
        w_ip = ip.weight      # Gewichtung
        J_ip = ip.jacobian    # Jakobische Determinante (Flächenelement)
        
        # Berechne Testfunktions-Werte an diesem Punkt
        # φ_i(x_ip) für alle DOFs i, die an diesem Facet beteiligt sind
        test_function_values = evaluate_test_functions(V, x_ip, facet_idx)
        
        # Korrekte Integration:
        # b_i += (∂p/∂n) * φ_i(x_ip) * w_ip * |J_ip|
        for dof_idx in facet_dofs[facet_idx]:
            phi_value = test_function_values[dof_idx]
            b.array[dof_idx] += neumann_value * phi_value * w_ip * J_ip
```

## Warum ist das wichtig?

### Beispiel:

Angenommen, wir haben ein Panel mit 3 DOFs auf einer Dreiecks-Facet:

```
DOF 1 (Ecke 1): φ_1(x_1) = 1.0, φ_1(x_2) = 0.0, φ_1(x_3) = 0.0
DOF 2 (Ecke 2): φ_2(x_1) = 0.0, φ_2(x_2) = 1.0, φ_2(x_3) = 0.0
DOF 3 (Ecke 3): φ_3(x_1) = 0.0, φ_3(x_2) = 0.0, φ_3(x_3) = 1.0
```

**Aktuelle Methode (falsch):**
```python
# Gleichmäßige Verteilung:
b[1] += neumann_value * (area / 3)  # φ_1 ≈ 1.0 an allen Punkten ❌
b[2] += neumann_value * (area / 3)
b[3] += neumann_value * (area / 3)
```

**Korrekte Methode:**
```python
# Integration über Facet mit Integration Points:
# Mittelpunkt des Dreiecks: x_mid
φ_1_mid = 0.33  # Tatsächlicher Wert von φ_1 am Mittelpunkt
φ_2_mid = 0.33
φ_3_mid = 0.33

# Korrekte Integration:
b[1] += neumann_value * φ_1_mid * weight * area_element
b[2] += neumann_value * φ_2_mid * weight * area_element
b[3] += neumann_value * φ_3_mid * weight * area_element
```

## Praktische Umsetzung in dolfinx

### Option 1: Facet-Tags verwenden (ideal, aber erfordert Panel-Flächen im Mesh)

```python
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector

# Panel-Fläche muss als Facet-Tag im Mesh existieren
ds_panel = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, tag=panel_tag)

# Korrekte UFL-Form:
neumann_form = ufl.inner(neumann_constant, test_function) * ds_panel
L_form = fem.form(L_form + neumann_form)

# dolfinx berechnet automatisch die korrekte Integration!
b = assemble_vector(L_form)
```

### Option 2: Manuelle Integration mit Integration Points (aktueller Fall)

```python
# Problem: Wir haben keine Facet-Tags, nur DOF-Koordinaten
# Lösung: Manuelle Integration über gefundene DOFs

from dolfinx import geometry
import numpy as np

# Finde Integration Points auf Panel-Fläche (z.B. über KD-Tree)
# Für jeden Integration Point:
for ip_coord in integration_points:
    # 1. Finde beteiligte DOFs (nächstgelegene DOFs)
    dof_indices = find_nearest_dofs(ip_coord, radius)
    
    # 2. Berechne Testfunktions-Werte φ_i(ip_coord)
    #    (erfordert Zugriff auf Finite-Element-Basis-Funktionen)
    test_values = evaluate_basis_functions(V, ip_coord, dof_indices)
    
    # 3. Berechne Flächenelement (Jakobische Determinante)
    area_element = compute_area_element(ip_coord, panel_normal)
    
    # 4. Korrekte Integration
    for i, dof_idx in enumerate(dof_indices):
        phi_i = test_values[i]
        b.array[dof_idx] += neumann_value * phi_i * weight * area_element
```

## Fazit

**Aktuelle Methode:**
- ✅ Einfach zu implementieren
- ✅ Funktioniert für feine Meshes
- ❌ Nur eine Approximation
- ❌ Ignoriert Testfunktions-Werte

**Exakte Methode:**
- ✅ Mathematisch korrekt
- ✅ Berücksichtigt Testfunktions-Werte
- ❌ Komplexer zu implementieren
- ❌ Erfordert Integration Points oder Facet-Tags

**Empfehlung:**
- Für die meisten Anwendungen ist die aktuelle Approximation ausreichend
- Für höchste Genauigkeit sollten Panel-Flächen als Facet-Tags im Mesh existieren
- Dann kann dolfinx automatisch die korrekte Integration durchführen


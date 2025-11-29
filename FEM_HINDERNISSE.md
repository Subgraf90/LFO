# Hindernisse in FEM-Berechnungen

## Übersicht

In FEM-Berechnungen werden Hindernisse (Surfaces) als **geometrische Objekte im Mesh** modelliert, die **Randbedingungen** auf ihren Oberflächen haben. Die Beugung (Diffraktion) entsteht automatisch durch die numerische Lösung der Helmholtz-Gleichung.

## Physikalische Grundlagen

### Wie funktioniert Beugung in FEM?

1. **Helmholtz-Gleichung**: Die Wellengleichung wird im gesamten Volumen gelöst
2. **Randbedingungen**: Hindernisse definieren Randbedingungen auf ihren Oberflächen
3. **Automatische Beugung**: Die numerische Lösung berücksichtigt automatisch, dass Wellen um Hindernisse herum gebeugt werden

### Vergleich: Superposition vs. FEM

| Aspekt | Superposition | FEM |
|--------|--------------|-----|
| **Schallausbreitung** | Geradlinig (wie Licht) | Wellenausbreitung mit Beugung |
| **Schatten** | Vollständiger Schatten | Teilweise Beugung um Hindernisse |
| **Hindernisse** | Ray-Tracing (Schatten-Maske) | Randbedingungen im Mesh |

## Modellierung von Hindernissen in FEM

### Ansatz 1: Dünne Platten als Volumen (Empfohlen)

Hindernisse werden als **dünne Volumen** im Mesh modelliert:

```python
# Pseudocode für GMSH-Mesh-Erstellung
# 1. Erstelle Haupt-Domain (Box)
box = factory.addBox(-half_w, -half_l, 0.0, width, length, height)

# 2. Für jedes Hindernis (Surface):
for surface in enabled_surfaces:
    # Erstelle dünne Platte als Volumen
    # z.B. 5-10 cm dick (abhängig von Frequenz)
    obstacle_thickness = 0.05  # 5 cm
    obstacle = factory.addBox(
        x_min, y_min, z_min,
        x_max - x_min, y_max - y_min, obstacle_thickness
    )
    
    # Subtrahiere Hindernis von Domain (Boolean-Operation)
    # → Erstellt Löcher im Mesh
    box = factory.cut([(3, box)], [(3, obstacle)])

# 3. Markiere Hindernis-Oberflächen als Physical Groups
# → Für Randbedingungen
```

**Vorteile**:
- Realistische Geometrie
- Automatische Beugung
- Korrekte Wellenausbreitung

**Nachteile**:
- Komplexere Mesh-Generierung
- Mehr DOFs (Freiheitsgrade)

### Ansatz 2: Randbedingungen auf internen Flächen

Hindernisse werden als **Flächen** im Mesh modelliert (kein Volumen):

```python
# Pseudocode
# 1. Erstelle Haupt-Domain
box = factory.addBox(...)

# 2. Für jedes Hindernis:
# Erstelle Fläche (kein Volumen)
obstacle_face = factory.addRectangle(x_min, y_min, z, width, height)

# 3. Markiere Fläche als Physical Group
obstacle_tag = gmsh.model.addPhysicalGroup(2, [obstacle_face], tag_id)

# 4. In FEM-Formulierung:
# Dirichlet-Randbedingung: p = 0 (harte, reflektierende Oberfläche)
# oder
# Neumann-Randbedingung: ∂p/∂n = 0 (weiche, reflektierende Oberfläche)
# oder
# Robin-Randbedingung: ∂p/∂n + ikαp = 0 (absorbierende Oberfläche)
```

**Vorteile**:
- Einfacher zu implementieren
- Weniger DOFs

**Nachteile**:
- Weniger realistisch (unendlich dünn)
- Kann numerische Probleme verursachen

### Ansatz 3: Materialeigenschaften (Erweitert)

Hindernisse haben **andere Materialeigenschaften** (z.B. sehr hohe Dichte):

```python
# Pseudocode
# 1. Erstelle Hindernis als Volumen mit Material-Tag
obstacle = factory.addBox(...)
obstacle_material_tag = gmsh.model.addPhysicalGroup(3, [obstacle], material_tag)

# 2. In FEM-Formulierung:
# Verschiedene Materialeigenschaften pro Zelle
# z.B. sehr hohe Dichte = undurchlässig
```

**Vorteile**:
- Sehr realistisch
- Kann verschiedene Materialien modellieren

**Nachteile**:
- Komplexeste Implementierung
- Benötigt Materialdatenbank

## Empfohlene Implementierung

### Schritt 1: Surface-zu-GMSH-Geometrie

```python
def _add_obstacles_to_gmsh_mesh(
    self,
    factory,
    enabled_surfaces: List[Tuple[str, Dict]],
    obstacle_thickness: float = 0.05,
) -> List[int]:
    """
    Fügt Hindernisse (Surfaces) als dünne Volumen zum GMSH-Mesh hinzu.
    
    Args:
        factory: GMSH factory (gmsh.model.occ)
        enabled_surfaces: Liste von (surface_id, surface_dict) Tupeln
        obstacle_thickness: Dicke der Hindernisse in Metern (Standard: 5 cm)
    
    Returns:
        Liste von GMSH-Volumen-Tags für Hindernisse
    """
    obstacle_tags = []
    
    for surface_id, surface_dict in enabled_surfaces:
        if not surface_dict.get("enabled", False):
            continue
            
        # Hole Surface-Punkte
        points = surface_dict.get("points", [])
        if len(points) < 3:
            continue
            
        # Konvertiere zu NumPy-Array
        coords = np.array([
            [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]
            for p in points
        ])
        
        # Berechne Bounding Box
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)
        
        # Erstelle dünne Platte als Volumen
        # Dicke in Z-Richtung (kann auch in andere Richtung sein)
        obstacle = factory.addBox(
            x_min, y_min, z_min,
            x_max - x_min,
            y_max - y_min,
            obstacle_thickness
        )
        
        obstacle_tags.append(obstacle)
        
    return obstacle_tags
```

### Schritt 2: Boolean-Operationen

```python
def _generate_gmsh_mesh_with_obstacles(
    self,
    width: float,
    length: float,
    height: float,
    resolution: float,
    enabled_surfaces: List[Tuple[str, Dict]],
) -> Tuple["mesh.Mesh", Optional["mesh.MeshTags"], Optional["mesh.MeshTags"]]:
    """
    Erstellt GMSH-Mesh mit Hindernissen.
    """
    factory = gmsh.model.occ
    
    # 1. Erstelle Haupt-Domain
    box = factory.addBox(-width/2, -length/2, 0.0, width, length, height)
    
    # 2. Füge Hindernisse hinzu
    obstacle_tags = self._add_obstacles_to_gmsh_mesh(
        factory, enabled_surfaces, obstacle_thickness=0.05
    )
    
    # 3. Subtrahiere Hindernisse von Domain
    if obstacle_tags:
        # Boolean-Operation: Domain - Hindernisse
        # → Erstellt Löcher im Mesh
        factory.cut([(3, box)], [(3, tag) for tag in obstacle_tags])
        factory.synchronize()
        
        # Markiere Hindernis-Oberflächen als Physical Groups
        # → Für Randbedingungen
        obstacle_surface_tag = 20  # Start-Tag für Hindernisse
        for i, tag in enumerate(obstacle_tags):
            # Hole Oberflächen des Hindernisses
            surfaces = gmsh.model.getBoundary([(3, tag)], oriented=False)
            if surfaces:
                phys_tag = obstacle_surface_tag + i
                gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], phys_tag)
                gmsh.model.setPhysicalName(2, phys_tag, f"obstacle_{i}")
    
    # 4. Generiere Mesh
    gmsh.model.mesh.generate(3)
    
    # ... Rest der Mesh-Erstellung
```

### Schritt 3: Randbedingungen auf Hindernissen

```python
def _build_obstacle_boundary_conditions(
    self,
    V: "fem.FunctionSpace",
    obstacle_tags: Dict[str, int],
    boundary_type: str = "dirichlet",  # "dirichlet", "neumann", "robin"
) -> List["fem.DirichletBC"]:
    """
    Erstellt Randbedingungen für Hindernisse.
    
    boundary_type:
        - "dirichlet": p = 0 (harte, reflektierende Oberfläche)
        - "neumann": ∂p/∂n = 0 (weiche, reflektierende Oberfläche)
        - "robin": ∂p/∂n + ikαp = 0 (absorbierende Oberfläche)
    """
    bcs = []
    
    if boundary_type == "dirichlet":
        # Dirichlet: p = 0 auf Hindernis-Oberflächen
        zero = fem.Constant(self._mesh, default_scalar_type(0.0))
        
        for obstacle_name, tag in obstacle_tags.items():
            # Finde Facets mit diesem Tag
            facets = np.where(self._facet_tags.values == tag)[0]
            if len(facets) > 0:
                bc = fem.dirichletbc(
                    zero,
                    fem.locate_dofs_topological(V, 2, facets),
                    V
                )
                bcs.append(bc)
    
    elif boundary_type == "robin":
        # Robin: Wird in der Formulierung behandelt (siehe _build_boundary_absorption_form)
        # → Keine separaten BCs nötig
        pass
    
    return bcs
```

### Schritt 4: Integration in Helmholtz-Formulierung

```python
def _solve_frequency(self, frequency: float) -> fem.Function:
    """Löst Helmholtz-Gleichung mit Hindernissen."""
    
    # ... bestehender Code ...
    
    # Erweitere Randbedingungen um Hindernisse
    obstacle_bcs = self._build_obstacle_boundary_conditions(
        V, self._obstacle_tags, boundary_type="dirichlet"
    )
    
    bcs = [] + obstacle_bcs  # Kombiniere mit anderen BCs
    
    # Löse System
    A = assemble_matrix(a_form, bcs=bcs)
    # ...
```

## Visualisierung der Beugung

Nach der FEM-Berechnung zeigt das Ergebnis automatisch Beugung:

1. **Schattenbereich**: Niedrigere Pegel (aber nicht Null!)
2. **Beugungszone**: Erhöhte Pegel um Hindernis-Kanten
3. **Interferenz**: Konstruktive/destruktive Interferenz hinter Hindernissen

## Zusammenfassung

| Schritt | Beschreibung |
|---------|-------------|
| **1. Geometrie** | Hindernisse als Volumen im GMSH-Mesh |
| **2. Boolean-Operationen** | Subtrahiere Hindernisse von Domain |
| **3. Randbedingungen** | Dirichlet/Neumann/Robin auf Hindernis-Oberflächen |
| **4. FEM-Lösung** | Helmholtz-Gleichung löst automatisch Beugung |
| **5. Ergebnis** | Schallfeld zeigt Beugung um Hindernisse |

**Wichtig**: Die Beugung entsteht **automatisch** durch die numerische Lösung - keine explizite Beugungsberechnung nötig!


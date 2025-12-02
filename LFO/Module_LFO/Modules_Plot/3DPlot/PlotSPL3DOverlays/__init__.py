"""3D-SPL-Plot Overlays Modul - Hauptklasse SPL3DOverlayRenderer.

Dieses Modul wurde vollständig refactored. Alle Funktionen sind in Submodulen
verfügbar. Die ursprüngliche SPL3DOverlayRenderer Klasse bleibt für
Rückwärtskompatibilität verfügbar.
"""

# Rückwärtskompatibilität: Original-Klassen bleiben verfügbar
try:
    from Module_LFO.Modules_Plot.PlotSPL3DOverlays import (  # noqa: F401
        SPL3DOverlayRenderer,
        SPLTimeControlBar,
    )
except ImportError:
    # Falls Original nicht verfügbar, verwende neue Implementierungen
    SPL3DOverlayRenderer = None  # noqa: F401
    SPLTimeControlBar = None  # noqa: F401

# Neue Module können bereits importiert werden:
from .impulse import (  # noqa: F401
    compute_impulse_state,
    draw_impulse_points,
    ImpulseRenderer,
)
from .ui_time_control import SPLTimeControlBar as NewSPLTimeControlBar  # noqa: F401
from .line_utils import LineUtils  # noqa: F401
from .caching import OverlayCaching  # noqa: F401
from .axis import (  # noqa: F401
    get_max_surface_dimension,
    get_active_xy_surfaces,
    get_surface_intersection_points_xz,
    get_surface_intersection_points_yz,
    draw_axis_planes,
)
from .surfaces import (  # noqa: F401
    compute_surfaces_signature,
    draw_surfaces,
)
from .speakers import (  # noqa: F401
    build_cabinet_lookup,
    get_speaker_info_from_actor,
    update_speaker_highlights,
    draw_speakers,
)
from .speaker_geometry import (  # noqa: F401
    get_speaker_name,
    safe_float,
    is_numeric,
    array_value,
    float_sequence,
    sequence_length,
    resolve_cabinet_entries,
    apply_flown_cabinet_shape,
    build_speaker_geometries,
)
from .renderer import (  # noqa: F401
    add_overlay_mesh,
    clear_overlays,
    clear_category,
)

__all__ = [
    # Hauptklasse
    'SPL3DOverlayRenderer',  # Original (temporär)
    'SPLTimeControlBar',  # Original (temporär)
    
    # Impulse
    'compute_impulse_state',
    'draw_impulse_points',
    'ImpulseRenderer',
    
    # UI Components
    'NewSPLTimeControlBar',
    
    # Line Utils
    'LineUtils',
    
    # Caching
    'OverlayCaching',
    
    # Axis
    'get_max_surface_dimension',
    'get_active_xy_surfaces',
    'get_surface_intersection_points_xz',
    'get_surface_intersection_points_yz',
    'draw_axis_planes',
    
    # Surfaces
    'compute_surfaces_signature',
    'draw_surfaces',
    
    # Speakers
    'build_cabinet_lookup',
    'get_speaker_info_from_actor',
    'update_speaker_highlights',
    'draw_speakers',
    
    # Speaker Geometry
    'get_speaker_name',
    'safe_float',
    'is_numeric',
    'array_value',
    'float_sequence',
    'sequence_length',
    'resolve_cabinet_entries',
    'apply_flown_cabinet_shape',
    'build_speaker_geometries',
    
    # Renderer
    'add_overlay_mesh',
    'clear_overlays',
    'clear_category',
]

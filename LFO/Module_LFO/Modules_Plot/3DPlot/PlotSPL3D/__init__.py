"""3D-SPL-Plot Modul - Hauptklasse DrawSPLPlot3D.

Dieses Modul wurde vollständig refactored. Die neue Hauptklasse ist DrawSPLPlot3DCore,
die alle Submodule koordiniert. Die ursprüngliche DrawSPLPlot3D Klasse bleibt für
Rückwärtskompatibilität verfügbar.
"""

# Neue Hauptklasse (vollständig migriert)
from .core import DrawSPLPlot3DCore  # noqa: F401

# Rückwärtskompatibilität: Original-Klasse bleibt verfügbar
try:
    from Module_LFO.Modules_Plot.PlotSPL3D import DrawSPLPlot3D  # noqa: F401
except ImportError:
    # Falls Original nicht verfügbar, verwende neue Klasse als Alias
    DrawSPLPlot3D = DrawSPLPlot3DCore  # noqa: F401

# Neue Module können bereits importiert werden:
from .utils import (  # noqa: F401
    has_valid_data,
    compute_surface_signature,
    quantize_to_steps,
    to_float_array,
    fallback_cmap_to_lut,
    compute_overlay_signatures,
)
from .interpolation import (  # noqa: F401
    bilinear_interpolate_grid,
    nearest_interpolate_grid,
)
from .camera import CameraManager  # noqa: F401
from .picking import PickingManager  # noqa: F401
from .colorbar import ColorbarManager, PHASE_CMAP  # noqa: F401
from .ui_components import (  # noqa: F401
    setup_view_controls,
    setup_time_control,
    create_view_button,
    create_axis_pixmap,
)
from .rendering import (  # noqa: F401
    update_surface_scalars,
    render_surfaces_textured,
    clear_vertical_spl_surfaces,
    get_vertical_color_limits,
    update_vertical_spl_surfaces,
)
from .event_handler import EventHandler  # noqa: F401

__all__ = [
    # Hauptklasse
    'DrawSPLPlot3DCore',  # Neue Hauptklasse (vollständig migriert)
    'DrawSPLPlot3D',  # Original (Rückwärtskompatibilität)
    
    # Utils
    'has_valid_data',
    'compute_surface_signature',
    'quantize_to_steps',
    'to_float_array',
    'fallback_cmap_to_lut',
    'compute_overlay_signatures',
    
    # Interpolation
    'bilinear_interpolate_grid',
    'nearest_interpolate_grid',
    
    # Manager
    'CameraManager',
    'PickingManager',
    'ColorbarManager',
    'EventHandler',
    
    # UI Components
    'setup_view_controls',
    'setup_time_control',
    'create_view_button',
    'create_axis_pixmap',
    
    # Rendering
    'update_surface_scalars',
    'render_surfaces_textured',
    'clear_vertical_spl_surfaces',
    'get_vertical_color_limits',
    'update_vertical_spl_surfaces',
    
    # Constants
    'PHASE_CMAP',
]

#!/bin/bash
# Skript zum Erstellen einer neuen virtuellen Umgebung für LFO

set -e  # Beende bei Fehlern

echo "=========================================="
echo "LFO Umgebung Setup"
echo "=========================================="
echo ""

# Farben für Output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Prüfe ob Conda/Mamba verfügbar ist
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo -e "${GREEN}✓ Mamba gefunden${NC}"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo -e "${YELLOW}⚠ Conda gefunden (Mamba wäre schneller)${NC}"
else
    echo -e "${RED}✗ Fehler: Weder Conda noch Mamba gefunden!${NC}"
    echo "Bitte installieren Sie Miniforge/Mambaforge:"
    echo "https://github.com/conda-forge/miniforge"
    exit 1
fi

echo ""
echo "Verwende: $CONDA_CMD"
echo ""

# Pfade definieren
PROJECT_ROOT="/Users/MGraf/Python/LFO_Umgebung"
NEW_ENV_PATH="$PROJECT_ROOT/Venv_FEM_new"
OLD_ENV_PATH="$PROJECT_ROOT/Venv_FEM"
BACKUP_PATH="$PROJECT_ROOT/Venv_FEM_backup_$(date +%Y%m%d_%H%M%S)"

# Wechsel ins Projektverzeichnis
cd "$PROJECT_ROOT"

# Prüfe ob alte Umgebung existiert
if [ -d "$OLD_ENV_PATH" ]; then
    echo -e "${YELLOW}Alte Umgebung gefunden: $OLD_ENV_PATH${NC}"
    read -p "Möchten Sie ein Backup erstellen? (j/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[JjYy]$ ]]; then
        echo "Erstelle Backup..."
        mv "$OLD_ENV_PATH" "$BACKUP_PATH"
        echo -e "${GREEN}✓ Backup erstellt: $BACKUP_PATH${NC}"
    else
        echo "Entferne alte Umgebung..."
        rm -rf "$OLD_ENV_PATH"
        echo -e "${GREEN}✓ Alte Umgebung entfernt${NC}"
    fi
fi

echo ""
echo "=========================================="
echo "Erstelle neue Conda-Umgebung..."
echo "=========================================="
echo ""

# Erstelle neue Umgebung
if [ -f "environment.yml" ]; then
    echo "Verwende environment.yml..."
    $CONDA_CMD env create -p "$NEW_ENV_PATH" -f environment.yml
else
    echo "Erstelle Basis-Umgebung..."
    $CONDA_CMD create -p "$NEW_ENV_PATH" python=3.11 -y
    
    echo "Installiere Pakete..."
    $CONDA_CMD install -p "$NEW_ENV_PATH" -c conda-forge \
        fenics-dolfinx \
        mpich \
        petsc \
        petsc4py \
        mpi4py \
        numpy \
        scipy \
        matplotlib \
        pyvista \
        pyvistaqt \
        pyqt=5 \
        qtpy \
        -y
fi

# Benenne neue Umgebung um
if [ -d "$NEW_ENV_PATH" ]; then
    mv "$NEW_ENV_PATH" "$OLD_ENV_PATH"
    echo -e "${GREEN}✓ Umgebung umbenannt zu: $OLD_ENV_PATH${NC}"
fi

echo ""
echo "=========================================="
echo "Installation abgeschlossen!"
echo "=========================================="
echo ""
echo "Aktivieren Sie die Umgebung mit:"
echo -e "${GREEN}conda activate $OLD_ENV_PATH${NC}"
echo ""
echo "Oder führen Sie Python direkt aus:"
echo -e "${GREEN}$OLD_ENV_PATH/bin/python${NC}"
echo ""

# Zeige installierte Pakete
echo "Installierte Hauptpakete:"
"$OLD_ENV_PATH/bin/python" -c "
import sys
packages = [
    'numpy', 'scipy', 'matplotlib', 
    'pyvista', 'PyQt5', 'qtpy',
    'dolfinx', 'mpi4py', 'petsc4py'
]
for pkg in packages:
    try:
        mod = __import__(pkg.replace('PyQt5', 'PyQt5.QtCore').split('.')[0])
        version = getattr(mod, '__version__', 'ok')
        print(f'  ✓ {pkg}: {version}')
    except ImportError:
        print(f'  ✗ {pkg}: NICHT INSTALLIERT')
"

echo ""
echo -e "${GREEN}Setup abgeschlossen!${NC}"


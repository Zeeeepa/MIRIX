#!/bin/bash
# Build script for Mirix client and server packages

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to restore pyproject.toml on exit
restore_pyproject() {
    if [ -f "pyproject.toml.backup" ]; then
        echo -e "${BLUE}Restoring pyproject.toml...${NC}"
        mv pyproject.toml.backup pyproject.toml
    fi
}

# Set trap to restore pyproject.toml on exit (success or failure)
trap restore_pyproject EXIT

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  Mirix Package Build Script${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Get script directory (scripts/packaging/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go up two levels: packaging/ -> scripts/ -> project_root/
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

# Clean previous builds
echo -e "${BLUE}[1/5] Cleaning previous builds...${NC}"
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -rf mirix_client.egg-info
rm -rf mirix_server.egg-info
rm -rf intuit_ecms_client.egg-info
rm -rf intuit_ecms_server.egg-info
rm -rf jl_ecms_client.egg-info
rm -rf jl_ecms_server.egg-info
echo -e "${GREEN}✓ Cleaned${NC}"
echo ""

# Temporarily rename pyproject.toml to prevent it from overriding setup scripts
echo -e "${BLUE}[1.5/5] Temporarily moving pyproject.toml...${NC}"
if [ -f "pyproject.toml" ]; then
    mv pyproject.toml pyproject.toml.backup
    echo -e "${GREEN}✓ Moved pyproject.toml${NC}"
else
    echo -e "${BLUE}  pyproject.toml already moved${NC}"
fi
echo ""

# Install build dependencies
echo -e "${BLUE}[2/5] Installing build dependencies...${NC}"
pip install --upgrade setuptools wheel twine
echo -e "${GREEN}✓ Build tools ready${NC}"
echo ""

# Build client package
echo -e "${BLUE}[3/5] Building jl-ecms-client package...${NC}"
python scripts/packaging/setup_client.py sdist bdist_wheel
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Client package built successfully${NC}"
    CLIENT_VERSION=$(ls dist/jl_ecms_client-*.whl 2>/dev/null | head -1 | sed -E 's/.*-([0-9]+\.[0-9]+\.[0-9]+)-.*/\1/')
    echo -e "   Version: ${GREEN}${CLIENT_VERSION}${NC}"
else
    echo -e "${RED}✗ Client package build failed${NC}"
    exit 1
fi
echo ""

# Build server package
echo -e "${BLUE}[4/5] Building jl-ecms-server package...${NC}"
python scripts/packaging/setup_server.py sdist bdist_wheel
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Server package built successfully${NC}"
    SERVER_VERSION=$(ls dist/jl_ecms_server-*.whl 2>/dev/null | head -1 | sed -E 's/.*-([0-9]+\.[0-9]+\.[0-9]+)-.*/\1/')
    echo -e "   Version: ${GREEN}${SERVER_VERSION}${NC}"
else
    echo -e "${RED}✗ Server package build failed${NC}"
    exit 1
fi
echo ""

# Summary
echo -e "${BLUE}[5/5] Build Summary${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "Built packages in: ${GREEN}dist/${NC}"
echo ""
echo -e "Client Package:"
ls -lh dist/jl_ecms_client-* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo -e "Server Package:"
ls -lh dist/jl_ecms_server-* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}✓ All packages built successfully!${NC}"
echo ""
echo "To install locally:"
echo -e "  ${BLUE}pip install dist/jl_ecms_client-${CLIENT_VERSION}-py3-none-any.whl${NC}"
echo -e "  ${BLUE}pip install dist/jl_ecms_server-${SERVER_VERSION}-py3-none-any.whl${NC}"
echo ""
echo "To publish to PyPI:"
echo -e "  ${BLUE}twine upload dist/jl_ecms_client-${CLIENT_VERSION}*${NC}"
echo -e "  ${BLUE}twine upload dist/jl_ecms_server-${SERVER_VERSION}*${NC}"
echo ""
echo "To test packages:"
echo -e "  ${BLUE}twine check dist/*${NC}"
echo ""

# Restore pyproject.toml
if [ -f "pyproject.toml.backup" ]; then
    mv pyproject.toml.backup pyproject.toml
    echo -e "${GREEN}✓ Restored pyproject.toml${NC}"
    echo ""
fi


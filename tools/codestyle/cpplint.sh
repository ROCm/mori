#!/bin/bash

set -e

# Fixed cpplint version
CPPLINT_VERSION="2.0.2"

# Function to check if cpplint is available
check_cpplint() {
    if command -v cpplint >/dev/null 2>&1; then
        echo "Found cpplint: $(which cpplint)"
        return 0
    else
        echo "cpplint not found"
        return 1
    fi
}

# Function to install cpplint using pip
install_cpplint() {
    echo "Attempting to install cpplint version ${CPPLINT_VERSION}..."

    # Check if python is available
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        echo "Error: Python not found. Please install Python first."
        exit 1
    fi

    echo "Using Python: $(which $PYTHON_CMD)"

    # Install cpplint with fixed version
    echo "Installing cpplint==${CPPLINT_VERSION}..."
    $PYTHON_CMD -m pip install cpplint==${CPPLINT_VERSION}

    if [ $? -eq 0 ]; then
        echo "Successfully installed cpplint ${CPPLINT_VERSION}"
    else
        echo "Failed to install cpplint"
        exit 1
    fi
}

# Main logic
if ! check_cpplint; then
    echo "cpplint not found, attempting to install..."
    install_cpplint

    # Verify installation
    if ! check_cpplint; then
        echo "Error: cpplint installation failed or not in PATH"
        exit 1
    fi
fi

# Get cpplint version for verification
echo "Using cpplint version:"
cpplint --version 2>/dev/null || echo "cpplint version information not available"

# Run cpplint with provided arguments
cpplint $@

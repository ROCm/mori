#!/bin/bash

set -e

# Fixed clang-format version
CLANG_FORMAT_VERSION="20.1.8"

# Function to check if clang-format is available
check_clang_format() {
    if command -v clang-format >/dev/null 2>&1; then
        echo "Found clang-format: $(which clang-format)"
        return 0
    else
        echo "clang-format not found"
        return 1
    fi
}

# Function to install clang-format using pip
install_clang_format() {
    echo "Attempting to install clang-format version ${CLANG_FORMAT_VERSION}..."

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

    # Install clang-format with fixed version
    echo "Installing clang-format==${CLANG_FORMAT_VERSION}..."
    $PYTHON_CMD -m pip install clang-format==${CLANG_FORMAT_VERSION}

    if [ $? -eq 0 ]; then
        echo "Successfully installed clang-format ${CLANG_FORMAT_VERSION}"
    else
        echo "Failed to install clang-format"
        exit 1
    fi
}

# Main logic
if ! check_clang_format; then
    echo "clang-format not found, attempting to install..."
    install_clang_format

    # Verify installation
    if ! check_clang_format; then
        echo "Error: clang-format installation failed or not in PATH"
        exit 1
    fi
fi

# Get clang-format version for verification
echo "Using clang-format version:"
clang-format --version

# Run clang-format with provided arguments
clang-format $@

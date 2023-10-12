#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Run this script at project root by ".linter.sh" before you commit.
echo "Running isort..."
isort -y -sp . --profile black

echo "Running black..."
black -l 80 .

echo "Running flake..."
flake8 .
#!/bin/bash
# Type checking script for Time Series App

PYTHONPATH=$(pwd) poetry run mypy --config-file mypy.ini "Time Series App.py"
exit $?

#!/bin/bash
cd "$(dirname "$0")"
source ../.venv/bin/activate
tensorboard --logdir=runs --host=127.0.0.1 --port=6006

#!/bin/sh
set -e

# SparkSage Docker entrypoint.
#
# The image runs as a non-root user (sparksage, uid 1001) for runtime safety.
# A bind-mounted data directory (-v "$PWD/data:/app/data") retains the host's
# ownership, which the sparksage user typically cannot write to -- causing
# sqlite3 "unable to open database file" on startup. When the container starts
# as root (the image default before privilege drop), fix the ownership of the
# data directory and then drop to the non-root user before exec'ing the app.
# When already non-root (e.g. `docker run --user sparksage`), just exec.

DATA_DIR="${SPARKSAGE_DATA_DIR:-/app/data}"

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$DATA_DIR"
    chown -R sparksage:sparksage "$DATA_DIR" 2>/dev/null || true
    exec gosu sparksage:sparksage "$@"
fi

exec "$@"

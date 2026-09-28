# Two stages so the runtime image carries no build toolchain and no test dependencies.
#
# CPU-only torch throughout: nothing in this repository needs CUDA, and the default PyPI
# index would add roughly 2.5 GB of nvidia wheels to the image for no benefit. To train on
# a GPU, build from a CUDA base image and drop the --index-url below.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Torch first and on its own layer: it is by far the largest dependency and it changes far
# less often than this package does.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .


FROM python:3.12-slim AS runtime

# Offline by default. Every model here is built from an architecture definition, so a
# container that reaches for the Hugging Face Hub is a container doing something
# unintended -- better to fail loudly than to download silently.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=4

RUN useradd --create-home --uid 10001 ftlab
WORKDIR /home/ftlab

COPY --from=builder /opt/venv /opt/venv
COPY --chown=ftlab:ftlab data ./data
COPY --chown=ftlab:ftlab docs ./docs
COPY --chown=ftlab:ftlab README.md LICENSE ./

# Created before the VOLUME declarations and owned by the unprivileged user: a volume
# mounted onto a directory that does not exist yet would be created root-owned, and the
# first write would fail.
RUN mkdir -p runs adapters reports && chown ftlab:ftlab runs adapters reports

USER ftlab

VOLUME ["/home/ftlab/runs", "/home/ftlab/adapters"]

ENTRYPOINT ["ftlab"]
CMD ["budget", "--r", "16", "--target", "attention"]

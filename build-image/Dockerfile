# FROM quay.io/modh/runtime-images@sha256:7dd23e58291cad7a0ab4a8e04bda06492f2c027eb33b226358380db58dcdd60b
FROM quay.io/modh/runtime-images@sha256:78154a777a1a0c292663ab13c7ec0f0beb5c5a6da54b1e63a519b6ecb703c644

COPY no-deps-req.txt ./
RUN pip install --no-deps --no-cache -r no-deps-req.txt && \
    pip uninstall -y opencv-python && \
    pip uninstall -y opencv-python-headless && \
    pip install opencv-python-headless==4.8.0.74
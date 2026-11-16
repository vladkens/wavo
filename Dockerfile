# wavo with Vulkan's loader and Mesa's drivers (Intel, AMD), from the binaries of `make docker-dist`.
FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers \
  && rm -rf /var/lib/apt/lists/*
ARG TARGETARCH
COPY --chmod=755 dist/linux-${TARGETARCH}/wavo /usr/local/bin/wavo
# NVIDIA's container toolkit mounts its Vulkan driver only with `graphics`.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
LABEL org.opencontainers.image.source=https://github.com/vladkens/wavo
ENTRYPOINT ["wavo"]

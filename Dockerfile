# wavo from `build`'s Linux binaries (the `docker` job puts them in dist/), with Mesa's Vulkan
# drivers for Intel and AMD GPUs; without a GPU wavo runs on the CPU.
FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers \
  && rm -rf /var/lib/apt/lists/*
# Not root: uid 1000, the models in /home/wavo/.cache (a named volume takes its owner from here).
RUN useradd --create-home --uid 1000 wavo && install -d -o wavo -g wavo /home/wavo/.cache
ARG TARGETARCH
COPY --chmod=755 dist/${TARGETARCH}/wavo /usr/local/bin/wavo
# NVIDIA's container toolkit mounts its Vulkan driver only with `graphics`.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
LABEL org.opencontainers.image.source=https://github.com/vladkens/wavo
USER wavo
ENTRYPOINT ["wavo"]

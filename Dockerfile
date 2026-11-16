# wavo from `make docker-dist`, with Vulkan's loader; on amd64 also Mesa's drivers for Intel and
# AMD. Debian's AMD driver links LLVM (150MB) for its fallback compiler, so it's built here
# without; Intel's needs no LLVM and comes from Debian's package. No llvmpipe: without a GPU,
# wavo runs on the CPU.

FROM debian:trixie-slim AS radv
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
  meson ninja-build gcc g++ pkg-config python3-mako python3-yaml python3-packaging bison flex \
  glslang-tools libdrm-dev libelf-dev libexpat1-dev zlib1g-dev libzstd-dev
ARG MESA=25.0.7
ARG MESA_SHA256=592272df3cf01e85e7db300c449df5061092574d099da275d19e97ef0510f8a6
RUN curl -sSfLo mesa.tar.xz https://archive.mesa3d.org/mesa-$MESA.tar.xz \
  && echo "$MESA_SHA256  mesa.tar.xz" | sha256sum -c && tar -xf mesa.tar.xz
RUN cd mesa-$MESA && meson setup build --prefix=/usr --libdir=lib/x86_64-linux-gnu \
  --buildtype=release -Db_ndebug=true -Dstrip=true -Dvulkan-drivers=amd -Dllvm=disabled \
  -Dgallium-drivers= -Dplatforms= -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dopengl=false \
  -Dgles1=disabled -Dgles2=disabled -Dvideo-codecs= -Dvulkan-layers= -Dtools= \
  && DESTDIR=/out ninja -C build install

FROM debian:trixie-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends libvulkan1 \
  && rm -rf /var/lib/apt/lists/*

# No GPU driver: Docker on a Mac or an ARM server runs on the CPU, or NVIDIA's through its toolkit.
FROM base AS arm64

FROM base AS amd64
RUN apt-get update && apt-get install -y --no-install-recommends libdrm2 libdrm-amdgpu1 \
  libelf1t64 libexpat1 libwayland-client0 libx11-xcb1 libxcb-dri3-0 libxcb-present0 \
  libxcb-randr0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxshmfence1 \
  && mkdir -p /usr/share/vulkan/icd.d && cd /tmp && apt-get download mesa-vulkan-drivers \
  && dpkg-deb -x mesa-vulkan-drivers_*.deb mesa \
  && cp mesa/usr/lib/x86_64-linux-gnu/libvulkan_intel.so /usr/lib/x86_64-linux-gnu/ \
  && cp mesa/usr/share/vulkan/icd.d/intel_icd.json /usr/share/vulkan/icd.d/ \
  && rm -rf /tmp/* /var/lib/apt/lists/*
COPY --from=radv /out/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so /usr/lib/x86_64-linux-gnu/
COPY --from=radv /out/usr/share/vulkan/icd.d/radeon_icd.x86_64.json /usr/share/vulkan/icd.d/

FROM ${TARGETARCH}
ARG TARGETARCH
COPY --chmod=755 dist/linux-${TARGETARCH}/wavo /usr/local/bin/wavo
# NVIDIA's container toolkit mounts its Vulkan driver only with `graphics`.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
LABEL org.opencontainers.image.source=https://github.com/vladkens/wavo
ENTRYPOINT ["wavo"]

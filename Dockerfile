# syntax=docker/dockerfile:1

FROM rust:1-slim-bookworm AS builder

WORKDIR /build

COPY . .

# Cache mounts keep the crate registry and the target/ dir between builds, so a
# source-only change no longer recompiles every dependency. The mounts are not
# part of the image layer, so the binary is copied out to /oxigate inside the
# same RUN — a later stage cannot COPY --from a cache mount.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/build/target,sharing=locked \
    cargo build --release --locked --bin oxigate \
    && cp /build/target/release/oxigate /oxigate

FROM gcr.io/distroless/cc-debian12:nonroot

COPY --from=builder /oxigate                   /oxigate
COPY --from=builder /build/config/oxigate.yaml /etc/oxigate/oxigate.yaml

# Licence texts travel with the binary, not just with the source tree. The bundled
# pricing dataset is MIT and is compiled in via include_bytes!, so its notice must
# accompany any binary built from it; the gateway itself is AGPL-3.0-or-later, which
# likewise requires the licence to be conveyed with the work.
COPY --from=builder /build/THIRD-PARTY-NOTICES.md /THIRD-PARTY-NOTICES.md
COPY --from=builder /build/LICENSE                /LICENSE

EXPOSE 8080

ENTRYPOINT ["/oxigate", "--config", "/etc/oxigate/oxigate.yaml"]

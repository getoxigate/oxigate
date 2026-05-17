FROM rust:1-slim-bookworm AS builder

WORKDIR /build

COPY . .

RUN cargo build --release --bin oxigate

FROM gcr.io/distroless/cc-debian12:nonroot

COPY --from=builder /build/target/release/oxigate    /oxigate
COPY --from=builder /build/config/oxigate.yaml       /etc/oxigate/oxigate.yaml

EXPOSE 8080

ENTRYPOINT ["/oxigate", "--config", "/etc/oxigate/oxigate.yaml"]

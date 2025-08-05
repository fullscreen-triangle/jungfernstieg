# Multi-stage Dockerfile for Jungfernstieg biological-virtual neural symbiosis system
# CRITICAL: This system manages living biological neural tissue
# All safety protocols must be validated before deployment

FROM rust:1.75-slim-bookworm AS builder

# Install system dependencies for biological hardware interfaces
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libudev-dev \
    libusb-1.0-0-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/jungfernstieg

# Copy workspace configuration
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ ./crates/

# Build the project with safety profile
RUN cargo build --profile safety --bin jungfernstieg-cli

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies for biological systems
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libudev1 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create user for safe operation (never run as root with biological systems)
RUN groupadd -r jungfernstieg && useradd -r -g jungfernstieg jungfernstieg

# Create necessary directories
RUN mkdir -p /opt/jungfernstieg/{bin,config,data,logs} \
    && chown -R jungfernstieg:jungfernstieg /opt/jungfernstieg

# Copy binary from builder
COPY --from=builder /usr/src/jungfernstieg/target/safety/jungfernstieg-cli /opt/jungfernstieg/bin/

# Copy configuration templates
COPY configs/ /opt/jungfernstieg/config/

# Set working directory and user
WORKDIR /opt/jungfernstieg
USER jungfernstieg

# Expose monitoring port
EXPOSE 8080

# Health check for biological system safety
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /opt/jungfernstieg/bin/jungfernstieg-cli safety --status || exit 1

# Environment variables
ENV RUST_LOG=info
ENV JUNGFERNSTIEG_CONFIG_PATH=/opt/jungfernstieg/config
ENV JUNGFERNSTIEG_DATA_PATH=/opt/jungfernstieg/data
ENV JUNGFERNSTIEG_LOG_PATH=/opt/jungfernstieg/logs

# CRITICAL SAFETY NOTICE
LABEL maintainer="Kundai Farai Sachikonye <kundai.sachikonye@wzw.tum.de>"
LABEL description="Jungfernstieg: Biological Neural Network Viability Through Virtual Blood Circulatory Systems"
LABEL version="0.1.0"
LABEL safety.level="BSL-2+"
LABEL biological.warning="MANAGES LIVING NEURAL TISSUE - SAFETY PROTOCOLS MANDATORY"
LABEL memorial.dedication="Saint Stella-Lorraine Masunda, Patron Saint of Impossibility"

# Default command - safety initialization required
ENTRYPOINT ["/opt/jungfernstieg/bin/jungfernstieg-cli"]
CMD ["safety", "--validate"]
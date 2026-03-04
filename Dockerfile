FROM bitnami/pytorch:latest

# Switch to root for installation
USER root

# Install system dependencies (Photon OS uses tdnf)
RUN tdnf install -y \
    build-essential \
    git \
    && tdnf clean all

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/

# Install the package with pip using system Python (which has PyTorch pre-installed)
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV POLARS_SKIP_CPU_CHECK=1

# Switch back to non-root user
USER 1001

# Default command
CMD ["bash"]

FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    git \
    ca-certificates \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Remove existing UID/GID 1000 ('ubuntu') and create 'vscode' with 1000:1000.
# No sudo: 'vscode' has no elevated privileges.
RUN userdel -r ubuntu 2>/dev/null || true \
    && groupdel ubuntu 2>/dev/null || true \
    && groupadd -g 1000 vscode \
    && useradd -m -u 1000 -g 1000 -s /bin/bash vscode

# Redirect all CUDA calls to CPU
ENV CUDA_VISIBLE_DEVICES=""

# Install Pixi system-wide into /usr/local/bin
ENV PIXI_BIN_DIR=/usr/local/bin
ENV PIXI_NO_PATH_UPDATE=1

RUN curl -fsSL https://pixi.sh/install.sh | bash \
    && mkdir -p /workspaces/mncm/.pixi \
    && chown -R vscode:vscode /workspaces/mncm \
    && echo 'eval "$(pixi completion -s bash)"' >> /etc/bash.bashrc

USER vscode
WORKDIR /workspaces/mncm

# Install OpenCode CLI for user 'vscode' and add to PATH
ENV OPENCODE_INSTALL_DIR=/usr/local/bin
RUN mkdir -p /home/vscode/.config/opencode \
    && chown -R vscode:vscode /home/vscode/.config
    
RUN curl -fsSL https://opencode.ai/v2/install | bash
ENV PATH="/home/vscode/.opencode/bin:${PATH}"

CMD ["pixi", "run", "--manifest-path", "pixi.toml", "python", "--help"]


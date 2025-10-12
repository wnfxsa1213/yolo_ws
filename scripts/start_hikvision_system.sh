#!/usr/bin/env bash
#
# 启动 Hikvision 混合架构全流程（容器服务 + 主程序）

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_BIN="$(command -v docker || true)"
if [ -z "${DOCKER_BIN}" ]; then
    echo "[start_hikvision] 未找到 docker，请在宿主机安装 Docker。" >&2
    exit 1
fi

log() {
    echo "[start_hikvision] $*"
}

# 1. 检查容器是否存在
CONTAINER="${HIKVISION_CONTAINER:-mvs-workspace}"
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    log "未找到容器 ${CONTAINER}，请先构建/启动 Hikvision docker 环境。"
    exit 1
fi

# 2. 启动容器内 camera_server（使用 supervisor 更稳）
log "启动容器内 camera_server 服务..."
docker exec "${CONTAINER}" bash -lc '
  set -e
  mkdir -p /workspace/logs
  export PYTHONPATH=/workspace/src:$PYTHONPATH
  source /etc/profile
  supervisorctl reread >/dev/null 2>&1 || true
  supervisorctl update >/dev/null 2>&1 || true
  supervisorctl restart camera_server >/dev/null 2>&1 || supervisorctl start camera_server >/dev/null 2>&1
'

# 3. 等待 socket 就绪
SOCKET="/tmp/hikvision.sock"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-10}"
log "等待 camera_server 就绪 (socket=${SOCKET})..."
elapsed=0
while (( elapsed < WAIT_TIMEOUT )); do
    if docker exec "${CONTAINER}" test -S "${SOCKET}"; then
        log "socket 已就绪。"
        break
    fi
    sleep 1
    ((elapsed++))
done
if (( elapsed >= WAIT_TIMEOUT )); then
    log "等待 camera_server 超时，请检查容器日志。"
    exit 1
fi

# 4. 如果本地已有旧 socket，优雅清理
HOST_SOCKET="${HOST_HIKVISION_SOCKET:-/tmp/hikvision.sock}"
if [ -S "${HOST_SOCKET}" ]; then
    log "删除宿主机旧 socket: ${HOST_SOCKET}"
    rm -f "${HOST_SOCKET}"
fi

# 5. 启动 socat，将容器内 UNIX socket 透出到宿主机
if ! command -v socat >/dev/null 2>&1; then
    log "未找到 socat，请先在宿主机安装（如 sudo apt-get install socat）。"
    exit 1
fi

log "启动本地 socat 转发 ${HOST_SOCKET} -> docker exec ${CONTAINER}:${SOCKET} ..."
PROXY_SCRIPT=$(mktemp)
cat <<EOF > "${PROXY_SCRIPT}"
#!/usr/bin/env bash
exec ${DOCKER_BIN} exec -i ${CONTAINER} socat STDIO UNIX-CONNECT:${SOCKET}
EOF
chmod +x "${PROXY_SCRIPT}"
set +e
socat UNIX-LISTEN:${HOST_SOCKET},fork,reuseaddr EXEC:"${PROXY_SCRIPT}" &
SOCAT_PID=$!
SOCAT_STATUS=$?
set -e
if [ ${SOCAT_STATUS} -ne 0 ]; then
    log "socat 启动失败，退出。"
    exit ${SOCAT_STATUS}
fi
trap 'log "清理 socat (${SOCAT_PID})"; kill ${SOCAT_PID} >/dev/null 2>&1 || true; rm -f "${HOST_SOCKET}" "${PROXY_SCRIPT}"' EXIT

# 4. 启动宿主机主程序
log "启动主程序 (config/system_config.yaml)..."
python3 "${PROJECT_ROOT}/src/main.py" --config "${PROJECT_ROOT}/config/system_config.yaml"
STATUS=$?
log "主程序退出状态: ${STATUS}"
exit ${STATUS}

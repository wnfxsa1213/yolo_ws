"""
网络参数调优工具。

按配置调用 ip/ethtool 等系统命令，失败时仅记录日志不中断主流程。
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Dict, Iterable, Optional


def _run_command(cmd: Iterable[str], logger: logging.Logger) -> None:
    argv = [str(part) for part in cmd]
    if not argv:
        return
    binary = argv[0]
    if shutil.which(binary) is None:
        logger.debug("命令 %s 不存在，跳过网络调优步骤。", binary)
        return
    try:
        completed = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - subprocess 异常
        logger.warning("执行命令失败 %s: %s", " ".join(argv), exc)
        return
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        logger.warning("命令 %s 失败 (code=%s): %s", " ".join(argv), completed.returncode, stderr)
    else:
        stdout = (completed.stdout or "").strip()
        if stdout:
            logger.debug("命令 %s 输出: %s", " ".join(argv), stdout)


def apply_network_tuning(config: Optional[Dict[str, object]], logger: logging.Logger) -> None:
    """
    根据配置应用常见的千兆网口优化参数。需要 root 权限，失败时仅记录日志。
    """

    if not config:
        return
    interface = config.get("interface")
    if not interface:
        logger.debug("网络调优配置缺少 interface 字段，跳过。")
        return

    logger.info("应用网络调优参数 iface=%s", interface)

    mtu = config.get("mtu")
    if mtu:
        _run_command(["ip", "link", "set", "dev", interface, "mtu", int(mtu)], logger)

    rx_ring = config.get("rx_ring")
    tx_ring = config.get("tx_ring")
    if rx_ring or tx_ring:
        args = ["ethtool", "-G", interface]
        if rx_ring:
            args.extend(["rx", str(int(rx_ring))])
        if tx_ring:
            args.extend(["tx", str(int(tx_ring))])
        _run_command(args, logger)

    def _apply_toggle(feature: str, enable: Optional[bool]) -> None:
        if enable is None:
            return
        state = "on" if enable else "off"
        _run_command(["ethtool", "-K", interface, feature, state], logger)

    _apply_toggle("gro", config.get("enable_gro"))
    _apply_toggle("lro", config.get("enable_lro"))
    _apply_toggle("ntuple", config.get("enable_ntuple"))

    if "rss" in config:
        enabled = bool(config.get("rss"))
        state = "on" if enabled else "off"
        _run_command(["ethtool", "-K", interface, "rxhash", state], logger)

    # 预留静态 IP 配置提示（实际视环境可能需要 netplan/NetworkManager）
    for key in ("host_ip", "camera_ip"):
        if config.get(key):
            logger.debug("网络调优提示: 需要保证 %s=%s 已通过系统网络配置生效。", key, config[key])

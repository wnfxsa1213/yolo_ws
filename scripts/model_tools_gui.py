#!/usr/bin/env python3
"""
老王专用GUI，把导出ONNX和构建TensorRT引擎的苦力活塞进一个窗口。
"""
from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 这破脚本要复用CLI逻辑，只能把scripts目录塞进sys.path，别嫌我粗暴。
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from build_engine import build_engine  # noqa: E402
from export_onnx import export as export_onnx_model  # noqa: E402


class ModelToolsGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("YOLO 模型工具 - 老王定制")
        self.root.geometry("760x620")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use("clam")

        self._build_export_section()
        self._build_engine_section()
        self._build_log_section()

    def _build_export_section(self) -> None:
        frame = ttk.LabelFrame(self.root, text="导出ONNX")
        frame.pack(fill=tk.X, padx=16, pady=(16, 8))

        self.weights_var = tk.StringVar(value=str(PROJECT_ROOT / "models/yolov8n.pt"))
        self.imgsz_var = tk.IntVar(value=640)
        self.output_var = tk.StringVar(value=str(PROJECT_ROOT / "models/yolov8n.onnx"))
        self.opset_var = tk.IntVar(value=12)
        self.dynamic_var = tk.BooleanVar(value=False)

        self._add_file_picker(frame, "权重(.pt)", self.weights_var, is_directory=False)
        self._add_spinbox(frame, "输入尺寸", self.imgsz_var, 32, 4096, 32)
        self._add_file_picker(frame, "输出ONNX", self.output_var, save_dialog=True)
        self._add_spinbox(frame, "ONNX Opset", self.opset_var, 11, 18, 1)

        dynamic_check = ttk.Checkbutton(frame, text="启用动态维度", variable=self.dynamic_var)
        dynamic_check.pack(anchor="w", padx=12, pady=4)

        export_btn = ttk.Button(frame, text="导出 ONNX", command=self._confirm_export)
        export_btn.pack(anchor="e", padx=12, pady=(4, 8))

    def _build_engine_section(self) -> None:
        frame = ttk.LabelFrame(self.root, text="构建TensorRT引擎")
        frame.pack(fill=tk.X, padx=16, pady=8)

        self.onnx_var = tk.StringVar(value=str(PROJECT_ROOT / "models/yolov8n.onnx"))
        self.engine_var = tk.StringVar(value=str(PROJECT_ROOT / "models/yolov8n_fp16.engine"))
        self.fp16_var = tk.BooleanVar(value=True)
        self.workspace_var = tk.IntVar(value=2)
        self.batch_var = tk.IntVar(value=1)
        self.height_var = tk.IntVar(value=640)
        self.width_var = tk.IntVar(value=640)

        self._add_file_picker(frame, "ONNX模型", self.onnx_var, is_directory=False)
        self._add_file_picker(frame, "输出引擎", self.engine_var, save_dialog=True)

        fp16_check = ttk.Checkbutton(frame, text="启用 FP16", variable=self.fp16_var)
        fp16_check.pack(anchor="w", padx=12, pady=(4, 0))

        self._add_spinbox(frame, "Workspace(GB)", self.workspace_var, 1, 16, 1)
        self._add_spinbox(frame, "Batch", self.batch_var, 1, 16, 1)
        self._add_spinbox(frame, "输入高度", self.height_var, 32, 4096, 32)
        self._add_spinbox(frame, "输入宽度", self.width_var, 32, 4096, 32)

        build_btn = ttk.Button(frame, text="构建 TensorRT", command=self._confirm_engine)
        build_btn.pack(anchor="e", padx=12, pady=(4, 8))

    def _build_log_section(self) -> None:
        frame = ttk.LabelFrame(self.root, text="操作日志")
        frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(8, 16))

        self.log_text = tk.Text(frame, height=14, wrap=tk.WORD, state=tk.DISABLED, background="#111", foreground="#0f0")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _add_file_picker(
        self,
        parent: ttk.LabelFrame,
        label: str,
        variable: tk.StringVar,
        *,
        save_dialog: bool = False,
        is_directory: bool = False,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=12, pady=4)

        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        def browse() -> None:
            initial = Path(variable.get()).resolve() if variable.get() else PROJECT_ROOT
            if is_directory:
                selected = filedialog.askdirectory(initialdir=str(initial))
            elif save_dialog:
                selected = filedialog.asksaveasfilename(initialdir=str(initial.parent), initialfile=initial.name)
            else:
                selected = filedialog.askopenfilename(initialdir=str(initial.parent))
            if selected:
                variable.set(selected)

        ttk.Button(row, text="浏览", command=browse).pack(side=tk.RIGHT)
        entry.insert(0, variable.get())

    def _add_spinbox(
        self, parent: ttk.LabelFrame, label: str, variable: tk.IntVar, minimum: int, maximum: int, step: int
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=12, pady=4)

        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        spin = ttk.Spinbox(row, from_=minimum, to=maximum, increment=step, textvariable=variable, width=8)
        spin.pack(side=tk.LEFT, padx=(0, 8))

    def _confirm_export(self) -> None:
        weights = Path(self.weights_var.get())
        if not weights.exists():
            messagebox.showerror("错误", f"找不到权重文件: {weights}")
            return
        threading.Thread(target=self._run_export, daemon=True).start()

    def _confirm_engine(self) -> None:
        onnx_path = Path(self.onnx_var.get())
        if not onnx_path.exists():
            messagebox.showerror("错误", f"找不到ONNX模型: {onnx_path}")
            return
        threading.Thread(target=self._run_engine, daemon=True).start()

    def _run_export(self) -> None:
        weights = Path(self.weights_var.get())
        output = Path(self.output_var.get())
        imgsz = self.imgsz_var.get()
        opset = self.opset_var.get()
        dynamic = self.dynamic_var.get()

        self._append_log(f"🚀 导出ONNX开始: {weights.name} -> {output.name}")
        try:
            export_onnx_model(weights, imgsz, output, opset, dynamic)
            self._append_log("✅ ONNX导出完成")
            messagebox.showinfo("完成", "ONNX导出搞定，别说我没帮你。")
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"❌ ONNX导出失败: {exc}")
            messagebox.showerror("导出失败", str(exc))

    def _run_engine(self) -> None:
        onnx_path = Path(self.onnx_var.get())
        engine_path = Path(self.engine_var.get())
        fp16 = self.fp16_var.get()
        workspace = self.workspace_var.get()
        batch = self.batch_var.get()
        height = self.height_var.get()
        width = self.width_var.get()

        self._append_log(f"⚙️ TensorRT构建开始: {onnx_path.name} -> {engine_path.name}")
        try:
            build_engine(onnx_path, engine_path, fp16, workspace, batch, height, width)
            self._append_log("✅ TensorRT引擎构建完成")
            messagebox.showinfo("完成", "TensorRT引擎构建完毕，拿去跑推理。")
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"❌ TensorRT构建失败: {exc}")
            messagebox.showerror("构建失败", str(exc))

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    gui = ModelToolsGUI()
    gui.run()


if __name__ == "__main__":
    main()

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSpinBox, QComboBox,
                             QPushButton, QSlider, QGroupBox)
from PyQt6.QtCore import QTimer, Qt
from conv_visualizer import ConvVisualizer3D
from conv_iterators import (
    TensorCoreOutputStationaryIterator,
    TensorCoreWeightStationaryIterator,
    TensorCoreInputStationaryIterator,
    TensorCoreSystolicIterator
)


class ConvMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Convolution on Tensor Core - Visualizer")
        self.resize(1600, 1000)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        # Default dimensions
        self.H = 6       # Input height
        self.W = 6       # Input width
        self.C_in = 4    # Input channels
        self.C_out = 4   # Output channels
        self.KH = 2      # Kernel height
        self.KW = 2      # Kernel width
        self.stride = 1
        self.padding = 0

        # Tensor Core dimensions
        self.TC_M = 8
        self.TC_N = 4
        self.TC_K = 8

        # State
        self.iterator = None
        self.generator = None
        self.is_running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_animation)

        self.setup_ui()
        self.init_visualizer()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QHBoxLayout(central_widget)

        # Left: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(340)
        self.layout.addWidget(control_panel)

        # Input Dimensions
        input_group = QGroupBox("Input Volume (H, W, C_in)")
        input_layout = QVBoxLayout()
        self.spin_h = self.create_spinbox("Input H:", self.H, input_layout)
        self.spin_w = self.create_spinbox("Input W:", self.W, input_layout)
        self.spin_cin = self.create_spinbox("Input C:", self.C_in, input_layout)
        input_group.setLayout(input_layout)
        control_layout.addWidget(input_group)

        # Kernel Dimensions
        kernel_group = QGroupBox("Kernel Volume (KH, KW, C_in, C_out)")
        kernel_layout = QVBoxLayout()
        self.spin_kh = self.create_spinbox("Kernel H:", self.KH, kernel_layout)
        self.spin_kw = self.create_spinbox("Kernel W:", self.KW, kernel_layout)
        self.spin_cout = self.create_spinbox("Output C:", self.C_out, kernel_layout)
        kernel_group.setLayout(kernel_layout)
        control_layout.addWidget(kernel_group)

        # Conv params
        param_group = QGroupBox("Convolution Params")
        param_layout = QVBoxLayout()
        self.spin_stride = self.create_spinbox("Stride:", self.stride, param_layout)
        self.spin_padding = self.create_spinbox("Padding:", self.padding, param_layout)
        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)

        # Tensor Core params
        tc_group = QGroupBox("Tensor Core Size")
        tc_layout = QVBoxLayout()
        self.spin_tc_m = self.create_spinbox("TC M:", self.TC_M, tc_layout)
        self.spin_tc_n = self.create_spinbox("TC N:", self.TC_N, tc_layout)
        self.spin_tc_k = self.create_spinbox("TC K:", self.TC_K, tc_layout)
        tc_group.setLayout(tc_layout)
        control_layout.addWidget(tc_group)

        # Algorithm Selection
        algo_group = QGroupBox("Tensor Core Dataflow")
        algo_layout = QVBoxLayout()

        self.combo_algo = QComboBox()
        self.combo_algo.addItems([
            "Output Stationary (OS) - Classic Tensor Core",
            "Weight Stationary (WS) - Kernel Fixed",
            "Input Stationary (IS) - Input Fixed",
            "Systolic Wavefront - Pipelined",
        ])
        algo_layout.addWidget(QLabel("Traversal Order:"))
        algo_layout.addWidget(self.combo_algo)

        algo_group.setLayout(algo_layout)
        control_layout.addWidget(algo_group)

        # Playback Controls
        play_group = QGroupBox("Controls")
        play_layout = QVBoxLayout()

        self.btn_start = QPushButton("Start / Pause")
        self.btn_start.clicked.connect(self.toggle_animation)
        play_layout.addWidget(self.btn_start)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_simulation)
        play_layout.addWidget(self.btn_reset)

        play_layout.addWidget(QLabel("Speed (ms delay):"))
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(10, 1500)
        self.slider_speed.setValue(500)
        self.slider_speed.valueChanged.connect(self.update_speed)
        play_layout.addWidget(self.slider_speed)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        play_layout.addWidget(self.lbl_status)

        # Stats Labels
        self.lbl_stats = QLabel("")
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setStyleSheet("color: green; font-weight: bold;")
        play_layout.addWidget(self.lbl_stats)

        # Info Labels
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: blue;")
        play_layout.addWidget(self.lbl_info)

        play_group.setLayout(play_layout)
        control_layout.addWidget(play_group)

        control_layout.addStretch()

        # Legend
        legend_group = QGroupBox("Legend")
        legend_layout = QVBoxLayout()
        legend_text = QLabel(
            "<b>Layout:</b><br>"
            "<span style='color:#993333;'>■ Input Volume</span> | "
            "<span style='color:#334D99;'>■ Kernel</span><br>"
            "<span style='color:#338080;'>■ Output</span> | "
            "<span style='color:#76C800;'>■ Tensor Core MAC</span><br><br>"
            "<b>Keys:</b><br>"
            "Space: Play/Pause<br>"
            "F: Step Forward<br>"
            "Mouse: Rotate/Zoom"
        )
        legend_layout.addWidget(legend_text)
        legend_group.setLayout(legend_layout)
        control_layout.addWidget(legend_group)

        # Right: Visualization
        self.viz_container = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_container)
        self.layout.addWidget(self.viz_container, stretch=1)

    def create_spinbox(self, label, initial, layout):
        l = QLabel(label)
        layout.addWidget(l)
        s = QSpinBox()
        s.setRange(1, 32)
        s.setValue(initial)
        s.valueChanged.connect(self.update_dims)
        layout.addWidget(s)
        return s

    def init_visualizer(self):
        if hasattr(self, 'visualizer'):
            self.viz_layout.removeWidget(self.visualizer)
            self.visualizer.deleteLater()

        self.H = self.spin_h.value()
        self.W = self.spin_w.value()
        self.C_in = self.spin_cin.value()
        self.C_out = self.spin_cout.value()
        self.KH = self.spin_kh.value()
        self.KW = self.spin_kw.value()
        self.stride = self.spin_stride.value()
        self.padding = self.spin_padding.value()
        self.TC_M = self.spin_tc_m.value()
        self.TC_N = self.spin_tc_n.value()
        self.TC_K = self.spin_tc_k.value()

        self.visualizer = ConvVisualizer3D(
            self.H, self.W, self.C_in, self.C_out,
            self.KH, self.KW, self.stride, self.padding,
            self.TC_M, self.TC_N, self.TC_K,
            key_event_callback=self.handle_visualizer_key
        )
        self.viz_layout.addWidget(self.visualizer)

        # Calculate and display dimensions
        OH = (self.H + 2 * self.padding - self.KH) // self.stride + 1
        OW = (self.W + 2 * self.padding - self.KW) // self.stride + 1
        M_total = OH * OW * self.C_out
        K_total = self.KH * self.KW * self.C_in
        total_ops = M_total * K_total

        self.lbl_info.setText(
            f"Output: {OH}x{OW}x{self.C_out}<br>"
            f"Output Space (M): {M_total}<br>"
            f"Reduction (K): {K_total}<br>"
            f"Total MACs: {total_ops}"
        )

        # Reset iterator
        self.iterator = None
        self.generator = None
        self.is_running = False
        self.timer.stop()
        self.lbl_status.setText("Ready")

    def update_dims(self):
        self.init_visualizer()

    def update_speed(self):
        if self.is_running:
            self.timer.setInterval(self.slider_speed.value())

    def toggle_animation(self):
        if self.is_running:
            self.timer.stop()
            self.is_running = False
            self.lbl_status.setText("Paused")
        else:
            if not self.generator:
                self.start_new_simulation()
            self.timer.start(self.slider_speed.value())
            self.is_running = True
            self.lbl_status.setText("Running...")

    def reset_simulation(self):
        self.timer.stop()
        self.is_running = False
        self.generator = None
        self.current_cycle = 0
        self.total_macs = 0
        self.visualizer.reset_simulation()
        self.lbl_status.setText("Reset")
        self.lbl_stats.setText("")

    def start_new_simulation(self):
        algo_text = self.combo_algo.currentText()

        self.visualizer.reset_simulation()

        self.current_cycle = 0
        self.total_macs = 0

        if "Output Stationary" in algo_text:
            self.iterator = TensorCoreOutputStationaryIterator(
                self.H, self.W, self.C_in, self.C_out,
                self.KH, self.KW, self.stride, self.padding,
                self.TC_M, self.TC_N, self.TC_K
            )
        elif "Weight Stationary" in algo_text:
            self.iterator = TensorCoreWeightStationaryIterator(
                self.H, self.W, self.C_in, self.C_out,
                self.KH, self.KW, self.stride, self.padding,
                self.TC_M, self.TC_N, self.TC_K
            )
        elif "Input Stationary" in algo_text:
            self.iterator = TensorCoreInputStationaryIterator(
                self.H, self.W, self.C_in, self.C_out,
                self.KH, self.KW, self.stride, self.padding,
                self.TC_M, self.TC_N, self.TC_K
            )
        elif "Systolic Wavefront" in algo_text:
            self.iterator = TensorCoreSystolicIterator(
                self.H, self.W, self.C_in, self.C_out,
                self.KH, self.KW, self.stride, self.padding,
                self.TC_M, self.TC_N, self.TC_K
            )

        self.generator = self.iterator.run()

    def step_animation(self):
        if not self.generator:
            return

        try:
            step = next(self.generator)

            if step.active_macs:
                self.current_cycle += 1
                macs_in_step = len(step.active_macs)
                self.total_macs += macs_in_step

                avg_ops = self.total_macs / self.current_cycle if self.current_cycle > 0 else 0

                stats_text = (
                    f"Cycle: {self.current_cycle}<br>"
                    f"MACs this step: {macs_in_step}<br>"
                    f"Total MACs: {self.total_macs}<br>"
                    f"Avg MACs/Cycle: {avg_ops:.2f}"
                )
                self.lbl_stats.setText(stats_text)

            self.visualizer.update_view(step)
            self.lbl_status.setText(step.description)
        except StopIteration:
            self.timer.stop()
            self.is_running = False
            self.lbl_status.setText("Finished")
            self.generator = None

    def next_frame(self):
        if not self.generator:
            self.start_new_simulation()
        if self.generator:
            self.step_animation()

    def handle_visualizer_key(self, event):
        self.keyPressEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.toggle_animation()
        elif key == Qt.Key.Key_F:
            if self.is_running:
                self.toggle_animation()
            self.next_frame()
        else:
            super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    window = ConvMainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

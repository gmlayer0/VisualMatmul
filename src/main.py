import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSpinBox, QComboBox, 
                             QPushButton, QSlider, QGroupBox)
from PyQt6.QtCore import QTimer, Qt
from visualizer import Visualizer3D
from iterators import NaiveIterator, TiledIterator, SystolicIterator, BlockedSystolicIterator, TensorSystolicIterator

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Matrix Multiplication Visualizer")
        self.resize(1200, 800)

        # Default Dims
        self.M = 24
        self.N = 24
        self.K = 24
        
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
        control_panel.setFixedWidth(300)
        self.layout.addWidget(control_panel)
        
        # Matrix Dimensions
        dim_group = QGroupBox("Dimensions")
        dim_layout = QVBoxLayout()
        
        self.spin_m = self.create_spinbox("M (Rows A):", self.M, dim_layout)
        self.spin_n = self.create_spinbox("N (Cols B):", self.N, dim_layout)
        self.spin_k = self.create_spinbox("K (Shared):", self.K, dim_layout)
        
        dim_group.setLayout(dim_layout)
        control_layout.addWidget(dim_group)
        
        # Algorithm Selection
        algo_group = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout()
        
        self.combo_algo = QComboBox()
        self.combo_algo.addItems([
            "Naive (ijk)", 
            "Naive (ikj)", 
            "Naive (jki)", 
            "Tensor Core (8x8x4)", 
            "Blocked Systolic (16x16 array, 1x1x1 pe)",
            "Tensor Core Systolic (4x4 array, 2x2x4 pe)"
        ])
        algo_layout.addWidget(QLabel("Type:"))
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
        self.slider_speed.setRange(10, 1000)
        self.slider_speed.setValue(100)
        self.slider_speed.valueChanged.connect(self.update_speed)
        play_layout.addWidget(self.slider_speed)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        play_layout.addWidget(self.lbl_status)
        
        play_group.setLayout(play_layout)
        control_layout.addWidget(play_group)
        
        control_layout.addStretch()
        
        # Right: Visualization Placeholder
        self.viz_container = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_container)
        self.layout.addWidget(self.viz_container, stretch=1)

    def create_spinbox(self, label, initial, layout):
        l = QLabel(label)
        layout.addWidget(l)
        s = QSpinBox()
        s.setRange(1, 512) # Increased limit
        s.setValue(initial)
        s.valueChanged.connect(self.update_dims)
        layout.addWidget(s)
        return s

    def init_visualizer(self):
        # Clear old
        if hasattr(self, 'visualizer'):
            self.viz_layout.removeWidget(self.visualizer)
            self.visualizer.deleteLater()
            
        self.M = self.spin_m.value()
        self.N = self.spin_n.value()
        self.K = self.spin_k.value()
        
        self.visualizer = Visualizer3D(self.M, self.N, self.K)
        self.viz_layout.addWidget(self.visualizer)
        
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
        self.visualizer.reset_simulation()
        self.lbl_status.setText("Reset")

    def start_new_simulation(self):
        self.visualizer.reset_simulation()
        algo_text = self.combo_algo.currentText()
        
        if "Naive" in algo_text:
            order = algo_text.split("(")[1].split(")")[0]
            self.iterator = NaiveIterator(self.M, self.N, self.K, order=order)
        elif "Tensor Core (8x8x4)" in algo_text:
            self.iterator = TiledIterator(self.M, self.N, self.K, tile_size=8, tile_k=4)
        # elif "Systolic Array" in algo_text:
        #     self.iterator = SystolicIterator(self.M, self.N, self.K)
        elif "Blocked Systolic" in algo_text:
            self.iterator = BlockedSystolicIterator(self.M, self.N, self.K, array_size=16)
        elif "Tensor Core Systolic" in algo_text:
            self.iterator = TensorSystolicIterator(self.M, self.N, self.K, array_size=4, micro_size=(2, 2, 4))
            
        self.generator = self.iterator.run()

    def step_animation(self):
        if not self.generator:
            return

        try:
            step = next(self.generator)
            self.visualizer.update_view(step.active, step.completed)
            self.lbl_status.setText(step.description)
        except StopIteration:
            self.timer.stop()
            self.is_running = False
            self.lbl_status.setText("Finished")
            self.generator = None

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()


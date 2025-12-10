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
        self.M = 12
        self.N = 12
        self.K = 12
        
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
            "Tensor Core (4x4x4)", 
            "Blocked Systolic (8x8 array, 1x1x1 pe)",
            "Tensor Core Systolic (2x2 array, 2x2x4 pe)"
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
        self.slider_speed.setValue(250)
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
        self.current_cycle = 0
        self.total_macs = 0
        self.visualizer.reset_simulation()
        self.lbl_status.setText("Reset")
        self.lbl_stats.setText("")

    def start_new_simulation(self):
        algo_text = self.combo_algo.currentText()
        
        # Determine systolic size for visualization
        systolic_size = None
        if "Blocked Systolic" in algo_text:
             systolic_size = 16 # Based on the label 16x16
        elif "Tensor Core Systolic" in algo_text:
             systolic_size = 4 # Based on 4x4 array
        elif "Tensor Core (8x8" in algo_text:
             systolic_size = 8 # Tiled acts like block
             
        self.visualizer.systolic_size = systolic_size
        self.visualizer.reset_simulation()
        
        self.current_cycle = 0
        self.total_macs = 0
        
        if "Naive" in algo_text:
            order = algo_text.split("(")[1].split(")")[0]
            self.iterator = NaiveIterator(self.M, self.N, self.K, order=order)
        elif "Tensor Core (4x4x4)" in algo_text:
            self.iterator = TiledIterator(self.M, self.N, self.K, tile_size=4, tile_k=4)
        # elif "Systolic Array" in algo_text:
        #     self.iterator = SystolicIterator(self.M, self.N, self.K)
        elif "Blocked Systolic" in algo_text:
            self.iterator = BlockedSystolicIterator(self.M, self.N, self.K, array_size=8)
        elif "Tensor Core Systolic" in algo_text:
            self.iterator = TensorSystolicIterator(self.M, self.N, self.K, array_size=2, micro_size=(2, 2, 4))
            
        self.generator = self.iterator.run()

    def step_animation(self):
        if not self.generator:
            return

        try:
            step = next(self.generator)
            
            # Stats calculation
            if step.active: # Only count cycles with activity (or count all steps?)
                # Assuming 1 step = 1 cycle in this simulation
                self.current_cycle += 1
                macs_in_step = len(step.active)
                self.total_macs += macs_in_step
                
                avg_ops = self.total_macs / self.current_cycle if self.current_cycle > 0 else 0
                
                stats_text = (
                    f"Cycle: {self.current_cycle}\n"
                    f"MACs this step: {macs_in_step}\n"
                    f"Avg MACs/Cycle: {avg_ops:.2f}"
                )
                self.lbl_stats.setText(stats_text)

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
    window.showMaximized()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()


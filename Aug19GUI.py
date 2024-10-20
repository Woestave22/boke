"""
Created on Sun Aug 18 23:52:01 2024
稳定版
@author: LaoWANG
"""


from Aug19PreProcessing import (detect_encoding, load_obj_file, load_mark_file, insert_midpoint_points, 
                                center_vertices, compute_bounding_box, extract_top_subcloud, find_best_rotation, 
                                flatten_and_analyze_curve, align_point_cloud, generate_topology)
import chardet
import csv
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import networkx as nx
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, 
                             QHBoxLayout, QLineEdit, QFileDialog, QSpacerItem, QSizePolicy)
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import sys



class TopologyEditor(QMainWindow):
    def __init__(self, marks2, marks2_file_name):
        super().__init__()
        self.file_name = marks2_file_name
        
        self.marks2 = np.array(marks2)
        tree = KDTree(self.marks2)
        self.G = generate_topology(self.marks2)
        self.high_degree_list = np.array([node for node in self.G.nodes() if self.G.degree(node) >= 3])
        self.high_degree_list = self.high_degree_list[np.argsort(self.high_degree_list)]
        self.counter_High_Degree = 0  # 节点中心化计数
        
        self.node_colors = ['r'] * len(self.marks2)  # Default color for nodes is red
        self.edge_colors = {edge: 'k' for edge in self.G.edges()}

        self.node1 = 0
        self.node_colors[self.node1] = 'b'
        self.node2 = None

        self.zoom_factor = 1.0
        self.zoom_step = 0.2
        self.indices_size = 15  # 设置indices的字体大小
        self.indices_distance = 0.1 * np.array([0,1,0]) # 设置indices到对应点的距离

        self.setFocusPolicy(Qt.StrongFocus)  # 允许接收键盘事件
        self.counter_Left_and_Right = None  # 方向左右键计数

        # Initial view parameters
        self.azim = -60
        self.elev = 30
        
        self.initUI()
    
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
    
        self.canvas = FigureCanvas(Figure(figsize=(10, 8)))
        # Set size policy to expanding
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        main_layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111, projection='3d')
        self.ax.view_init(elev=self.elev, azim=self.azim)
        
        # Create a widget for the input fields and buttons
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setAlignment(Qt.AlignCenter)
        control_layout.setSpacing(0)  # 设置控件之间的行距为最小
        
        # Create layouts for each row
        row0_layout = QHBoxLayout()
        row1_layout = QHBoxLayout()
        row2_layout = QHBoxLayout()
        row3_layout = QHBoxLayout()
        
        # 创建一个固定宽度的外部容器
        row0_container = QWidget()
        row0_container.setFixedWidth(1000)   # 设定行的固定宽度
        row0_container.setLayout(row0_layout)
        row1_container = QWidget()
        row1_container.setFixedWidth(1000)
        row1_container.setLayout(row1_layout)
        row2_container = QWidget()
        row2_container.setFixedWidth(1000)
        row2_container.setLayout(row2_layout)
        row3_container = QWidget()
        row3_container.setFixedWidth(1000)
        row3_container.setLayout(row3_layout)
        
        left_spacer = QSpacerItem(7, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)  # 创建占位符
        self.status_label = QLabel('Enter two node indices to add or delete an edge.')
        middle_spacer1 = QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.high_degree_count = QLabel('High degree nodes counting.')
        middle_spacer2 = QSpacerItem(12, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.center_next_button = QPushButton('Center on Next HDN')
        right_spacer = QSpacerItem(7, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.center_next_button.setFixedWidth(200)  # Set fixed width for consistency
        
        self.node1_input = QLineEdit()
        self.node1_input.setPlaceholderText('Node 1')
        self.node1_confirm_button = QPushButton('Confirm Node 1')
        self.add_edge_button = QPushButton('Add Edge')
        self.center_node_button = QPushButton('Center on Node 1')
        self.node2_input = QLineEdit()
        self.node2_input.setPlaceholderText('Node 2')
        self.node2_confirm_button = QPushButton('Confirm Node 2')
        self.delete_edge_button = QPushButton('Delete Edge')
        self.reset_view_button = QPushButton('Reset View')
        self.node1_input.setFixedWidth(470)  # Set fixed width for consistency
        self.node1_confirm_button.setFixedWidth(150)  # Set fixed width for consistency
        self.add_edge_button.setFixedWidth(150)  # Set fixed width for consistency
        self.center_node_button.setFixedWidth(150)  # Set fixed width for consistency
        self.node2_input.setFixedWidth(470)  # Set fixed width for consistency
        self.node2_confirm_button.setFixedWidth(150)  # Set fixed width for consistency
        self.delete_edge_button.setFixedWidth(150)  # Set fixed width for consistency
        self.reset_view_button.setFixedWidth(150)  # Set fixed width for consistency      

        # Create a widget for zoom and view controls
        zoom_and_view_widget = QWidget()
    
        self.zoom_in_button = QPushButton('Zoom In')
        self.zoom_out_button = QPushButton('Zoom Out')
        self.reset_zoom_button = QPushButton('Reset Zoom')
        self.save_button = QPushButton('Save Data')
        self.load_button = QPushButton('Load Data')
        self.close_button = QPushButton('Close') 
        self.zoom_in_button.setFixedWidth(180)  # Set fixed width for consistency
        self.zoom_out_button.setFixedWidth(180)  # Set fixed width for consistency
        self.reset_zoom_button.setFixedWidth(180)  # Set fixed width for consistency
        self.save_button.setFixedWidth(180)  # Set fixed width for consistency
        self.load_button.setFixedWidth(180)  # Set fixed width for consistency
        self.close_button.setFixedWidth(150)  # Set fixed width for consistency
        
        row0_layout.addSpacerItem(left_spacer)   # 用占位符将status label向右推，实现对齐
        row0_layout.addWidget(self.status_label)
        row0_layout.addSpacerItem(middle_spacer1) # 隔开
        row0_layout.addWidget(self.high_degree_count)
        row0_layout.addSpacerItem(middle_spacer2) # 隔开
        row0_layout.addWidget(self.center_next_button)
        row0_layout.addSpacerItem(right_spacer)  # 向左推
        row1_layout.addWidget(self.node1_input)
        row1_layout.addWidget(self.node1_confirm_button)
        row1_layout.addWidget(self.add_edge_button)
        row1_layout.addWidget(self.center_node_button)
        row2_layout.addWidget(self.node2_input)
        row2_layout.addWidget(self.node2_confirm_button)
        row2_layout.addWidget(self.delete_edge_button)
        row2_layout.addWidget(self.reset_view_button)
        row3_layout.addWidget(self.save_button)
        row3_layout.addWidget(self.load_button)
        # row3_layout.addWidget(self.close_button)
        row3_layout.addWidget(self.zoom_in_button)
        row3_layout.addWidget(self.zoom_out_button)
        row3_layout.addWidget(self.reset_zoom_button)

        # 将4行的容器添加到主布局中
        control_layout.addWidget(row0_container)        
        control_layout.addWidget(row1_container)
        control_layout.addWidget(row2_container)
        control_layout.addWidget(row3_container)

        main_layout.addWidget(control_widget)
        main_layout.addWidget(zoom_and_view_widget)
    
        # Store the initial view limits
        self.initial_xlim = (self.marks2[:, 0].min() - 1, self.marks2[:, 0].max() + 1)
        self.initial_ylim = (self.marks2[:, 1].min() - 1, self.marks2[:, 1].max() + 1)
        self.initial_zlim = (self.marks2[:, 2].min() - 1, self.marks2[:, 2].max() + 1)
    
        # Initialize current view limits and center point
        self.current_xlim = list(self.initial_xlim)
        self.current_ylim = list(self.initial_ylim)
        self.current_zlim = list(self.initial_zlim)
        self.current_center = [
            (self.initial_xlim[0] + self.initial_xlim[1]) / 2,
            (self.initial_ylim[0] + self.initial_ylim[1]) / 2,
            (self.initial_zlim[0] + self.initial_zlim[1]) / 2
        ]
    
        self.plot_graph()

        # Connect buttons to their functions
        self.node1_confirm_button.clicked.connect(lambda: self.confirm_node(1))
        self.node2_confirm_button.clicked.connect(lambda: self.confirm_node(2))
        self.add_edge_button.clicked.connect(self.add_edge)
        self.delete_edge_button.clicked.connect(self.delete_edge)
        self.center_node_button.clicked.connect(self.center_on_node1)
        self.center_next_button.clicked.connect(self.move_center_to_next_node)
        self.reset_view_button.clicked.connect(self.reset_view)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        self.save_button.clicked.connect(self.save_data)
        self.load_button.clicked.connect(self.load_data)
        # self.close_button.clicked.connect(self.close)

    def plot_graph(self):
        self.ax.clear()
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        
        # 计算当前视图范围
        x_min, x_max = self.current_xlim
        y_min, y_max = self.current_ylim
        z_min, z_max = self.current_zlim
    
        # 过滤在视图范围内的点
        mask = (
            (self.marks2[:, 0] >= x_min) & (self.marks2[:, 0] <= x_max) &
            (self.marks2[:, 1] >= y_min) & (self.marks2[:, 1] <= y_max) &
            (self.marks2[:, 2] >= z_min) & (self.marks2[:, 2] <= z_max)
        )
        filtered_indices = np.where(mask)[0]  # Extract the indices from the tuple
        
        # 绘制边
        for edge, color in self.edge_colors.items():
            if edge[0] in filtered_indices and edge[1] in filtered_indices:
                x = [self.marks2[edge[0], 0], self.marks2[edge[1], 0]]
                y = [self.marks2[edge[0], 1], self.marks2[edge[1], 1]]
                z = [self.marks2[edge[0], 2], self.marks2[edge[1], 2]]
                self.ax.plot(x, y, z, color=color)

        # Step 1: Calculate neighborhood radius
        neighborhood_radius = 2/ self.zoom_factor 
    
        # Step 2: Compute point cloud density
        tree = KDTree(self.marks2)
        densities = np.zeros(len(self.marks2))
    
        for i, mark in enumerate(self.marks2):
            distances, _ = tree.query(mark.reshape(1, -1), k=len(self.marks2))
            densities[i] = np.sum(distances[0] <= neighborhood_radius) - 1  # Exclude the point itself
    
        # Normalize densities to [0, 1]
        for i in range(len(densities)):
            densities[i] = min(1, densities[i] /5)

        # 定义节点
        self.high_degree_nodes = [node for node in self.G.nodes() if self.G.degree(node) >= 3]
        self.high_degree_count.setText(f'High degree nodes counting: {len(self.high_degree_nodes)}')
        
        # Step 3: Determine display probability and plot nodes
        for i in filtered_indices:
            self.node_colors[i] = 'limegreen' if i in self.high_degree_nodes and self.node_colors[i] == 'r' else self.node_colors[i]
            x, y, z = self.marks2[i]
            self.ax.scatter(x, y, z, c=self.node_colors[i], s=50)
    
            # Determine if label should be displayed based on density
            display_probability = self.zoom_factor - densities[i]  # Higher density -> lower probability, Bigger zoom factor -> Higher probability
            if np.random.rand() < display_probability:
                self.ax.text(x + self.indices_distance[0], y + self.indices_distance[1], z + self.indices_distance[2], str(i), color='k', fontsize=self.indices_size)

        # Additional functionality: Display nearest neighbors of node1
        if self.node1 is not None:
            # Find the nearest 10 neighbors of node1
            distances, indices = tree.query(self.marks2[self.node1].reshape(1, -1), k=11)
            nearest_indices = indices[0][0:]  # 含自身

            # Plot nearest neighbors
            for idx in nearest_indices:
                if idx in filtered_indices:
                    color = 'b' if idx == nearest_indices[0] else 'k'
                    mark = self.marks2[idx]
                    self.ax.text(mark[0] + self.indices_distance[0], mark[1] + self.indices_distance[1], mark[2] + self.indices_distance[2], str(idx), color=color, fontsize=self.indices_size)
        
        # 显示node2的index
        if self.node2 is not None:
            mark = self.marks2[self.node2]
            self.ax.text(mark[0] + self.indices_distance[0], mark[1] + self.indices_distance[1], mark[2] + self.indices_distance[2], str(self.node2), color='b', fontsize=self.indices_size)

        # 节点统一显示indices
        for idx in self.high_degree_nodes:
            if idx in filtered_indices:
                mark = self.marks2[idx]
                self.ax.text(mark[0] + self.indices_distance[0], mark[1] + self.indices_distance[1], mark[2] + self.indices_distance[2], str(idx), color='r', fontsize=self.indices_size)
            
        self.ax.set_box_aspect([1, 1, 1])
    
        # Apply the current view limits
        self.ax.set_xlim(self.current_xlim)
        self.ax.set_ylim(self.current_ylim)
        self.ax.set_zlim(self.current_zlim)

        self.node1_input.setText(str(self.node1))
        self.node2_input.setText(str(self.node2))
        
        self.canvas.draw()

    def confirm_node(self, node_number):
        try:
            node_index = int(self.node1_input.text() if node_number == 1 else self.node2_input.text())
            if 0 <= node_index < len(self.marks2):
                if node_number == 1:
                    if self.node1 is not None:
                        self.node_colors[self.node1] = 'r'  # Reset previous node1 color to red
                    self.node1 = node_index
                    self.node_colors[self.node1] = 'b'  # Set new node1 color to blue
                    self.counter_Left_and_Right = None  # 重置左右计数
                else:
                    if self.node2 is not None:
                        self.node_colors[self.node2] = 'r'  # Reset previous node2 color to red
                    self.node2 = node_index
                    self.node_colors[self.node2] = 'b'  # Set new node2 color to blue
                
                self.plot_graph()
            else:
                self.status_label.setText("Invalid node index.")
        except ValueError:
            self.status_label.setText("Please enter a valid integer.")

    def add_edge(self):
        try:
            if self.node1 is not None and self.node2 is not None:
                if self.node1 == self.node2:
                    self.status_label.setText("Both nodes are the same node.")
                elif not self.G.has_edge(self.node1, self.node2):
                    self.G.add_edge(self.node1, self.node2)
                    # Add new edge color
                    self.edge_colors[(self.node1, self.node2)] = 'b'
                    self.edge_colors[(self.node2, self.node1)] = 'b'
                    self.status_label.setText(f"Added edge between Node {self.node1} and Node {self.node2}.")
                else:
                    self.status_label.setText("Edge already exists.")
                                
                self.plot_graph()
            else:
                self.status_label.setText("Please confirm both node indices.")

        except ValueError:
            self.status_label.setText("Please enter valid integers for node indices.")

    def delete_edge(self):
        try:
            if self.node1 is not None and self.node2 is not None:
                if self.G.has_edge(self.node1, self.node2):
                    self.G.remove_edge(self.node1, self.node2)
                    # Remove edge color from the edge_colors dictionary
                    self.edge_colors.pop((self.node1, self.node2), None)
                    self.edge_colors.pop((self.node2, self.node1), None)
                    self.status_label.setText(f"Deleted edge between Node {self.node1} and Node {self.node2}.")
                else:
                    self.status_label.setText("Edge does not exist.")

                self.plot_graph()
            else:
                self.status_label.setText("Please confirm both node indices.")

        except ValueError:
            self.status_label.setText("Please enter valid integers for node indices.")

    def center_on_node1(self):
        if self.node1 is not None:
            x, y, z = self.marks2[self.node1]
            self.current_center = [x, y, z]
            self.update_zoom()  # Update zoom based on the new center
        else:
            self.status_label.setText("Please confirm Node 1 first.")

    def move_center_to_next_node(self):
        if self.node1 is not None:
            self.node_colors[self.node1] = 'r' if self.node1 != self.node2 else 'b'  # Set back old node1 color
            self.counter_High_Degree +=1
            if self.counter_High_Degree == len(self.high_degree_list):
                self.counter_High_Degree =0
                self.high_degree_list = self.high_degree_nodes[np.argsort(self.high_degree_nodes)]

            self.node1 = self.high_degree_list[self.counter_High_Degree]
            self.node_colors[self.node1] = 'b'  # Set new node1 color to blue
            x, y, z = self.marks2[self.node1]
            self.current_center = [x, y, z]
            self.update_zoom()
            self.counter_Left_and_Right = None  # 重置左右计数
        else:
            self.status_label.setText("Please confirm Node 1 first.")

    def reset_view(self):
        # Reset to initial view limits but keep the current zoom factor
        self.current_xlim = list(self.initial_xlim)
        self.current_ylim = list(self.initial_ylim)
        self.current_zlim = list(self.initial_zlim)
        self.current_center = [
            (self.current_xlim[0] + self.current_xlim[1]) / 2,
            (self.current_ylim[0] + self.current_ylim[1]) / 2,
            (self.current_zlim[0] + self.current_zlim[1]) / 2
        ]
        self.update_zoom()

    def zoom_in(self):
        self.zoom_factor += self.zoom_step
        self.update_zoom()

    def zoom_out(self):
        self.zoom_factor = max(0.1, self.zoom_factor - self.zoom_step)
        self.update_zoom()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.update_zoom()

    def update_zoom(self):
        x_range = (self.initial_xlim[1] - self.initial_xlim[0]) / self.zoom_factor
        y_range = (self.initial_ylim[1] - self.initial_ylim[0]) / self.zoom_factor
        z_range = (self.initial_zlim[1] - self.initial_zlim[0]) / self.zoom_factor

        # Calculate new limits centered on the current center
        self.current_xlim = [self.current_center[0] - x_range / 2, self.current_center[0] + x_range / 2]
        self.current_ylim = [self.current_center[1] - y_range / 2, self.current_center[1] + y_range / 2]
        self.current_zlim = [self.current_center[2] - z_range / 2, self.current_center[2] + z_range / 2]

        self.plot_graph()  

    def save_data(self):
        # 如果 self.file_name 后缀为 '.mark'，将其改为 '.csv'
        if self.file_name.endswith('.mark'):
            self.default_file_name = self.file_name.replace('.mark', '.csv')
        else:
            self.default_file_name = self.file_name
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", self.default_file_name, "CSV Files (*.csv);;All Files (*)", options=options)
        
        if file_path:
            # Save all data in a single CSV file
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['type', 'id1', 'id2', 'x', 'y', 'z', 'color'])
                # Save nodes
                for i, (x, y, z) in enumerate(self.marks2):
                    writer.writerow(['node', i, '', x, y, z, self.node_colors[i]])
                # Save edges
                for edge in self.G.edges():
                    writer.writerow(['edge', edge[0], edge[1], '', '', '', self.edge_colors.get((edge[0], edge[1]), 'k')])
            
            self.status_label.setText(f"Data saved to {file_path}.")

    def load_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Data", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            # Load all data from a single CSV file
            self.marks2 = []
            self.node_colors = []
            self.G = nx.Graph()
            self.edge_colors = {}

            # 获取文件名并设置为窗口标题
            self.file_name = os.path.basename(file_path)
            self.setWindowTitle(f"3D Graph - {self.file_name}")
            
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    data_type = row[0]
                    if data_type == 'node':
                        i, x, y, z, color = int(row[1]), float(row[3]), float(row[4]), float(row[5]), row[6]
                        self.marks2.append([x, y, z])
                        self.node_colors.append(color)
                    elif data_type == 'edge':
                        source, target, color = int(row[1]), int(row[2]), row[6]
                        self.G.add_edge(source, target)
                        self.edge_colors[(source, target)] = color
                        self.edge_colors[(target, source)] = color  # Ensure bidirectional edges have colors

            self.marks2 = np.array(self.marks2)
            self.node1 = 0
            self.node_colors[self.node1] = 'b'
            self.node2 = None
            self.high_degree_list = np.array([node for node in self.G.nodes() if self.G.degree(node) >= 3])
            self.high_degree_list = self.high_degree_list[np.argsort(self.high_degree_list)]
            self.counter_High_Degree = 0  # 节点中心化计数
            self.counter_Left_and_Right = None  # 方向左右键计数
    
            self.azim = -60
            self.elev = 30
            self.ax.view_init(elev=self.elev, azim=self.azim)
            self.zoom_factor = 1.0
            self.initial_xlim = (self.marks2[:, 0].min() - 1, self.marks2[:, 0].max() + 1)
            self.initial_ylim = (self.marks2[:, 1].min() - 1, self.marks2[:, 1].max() + 1)
            self.initial_zlim = (self.marks2[:, 2].min() - 1, self.marks2[:, 2].max() + 1)
            self.reset_view()
            self.status_label.setText(f"Data loaded from {file_path}.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.add_edge()
        elif event.key() == Qt.Key_Backspace:
            self.delete_edge()
        elif event.key() == Qt.Key_Shift:
            self.center_on_node1()
        elif event.key() == Qt.Key_Control:
            self.reset_view()
            
        elif event.key() == Qt.Key_J:
            self.zoom_in()
        elif event.key() == Qt.Key_K:
            self.zoom_out()
        elif event.key() == Qt.Key_L:
            self.reset_zoom()

        elif event.key() == Qt.Key_BracketLeft:
            self.load_data()
        elif event.key() == Qt.Key_BracketRight:
            self.save_data()

        # Adjust azimuth (rotation around z-axis)
        if event.key() == Qt.Key_U:
            self.azim -= 10
            self.ax.view_init(elev=self.elev, azim=self.azim)
            self.plot_graph()
        elif event.key() == Qt.Key_I:
            self.azim += 10
            self.ax.view_init(elev=self.elev, azim=self.azim)
            self.plot_graph()
        # Adjust elevation (rotation around x-axis)
        elif event.key() == Qt.Key_Y:
            self.elev += 10
            self.ax.view_init(elev=self.elev, azim=self.azim)
            self.plot_graph()
        elif event.key() == Qt.Key_H:
            self.elev -= 10
            self.ax.view_init(elev=self.elev, azim=self.azim)
            self.plot_graph()
        
        if self.node1 is not None:
            if event.key() == Qt.Key_Up or event.key() == Qt.Key_W:
                self.node_colors[self.node1] = 'r'
                if self.node2 is not None:
                    self.node_colors[self.node2] = 'b' 
                self.node1 = (self.node1 +1) % len(self.marks2)
                self.counter_Left_and_Right = None
                self.node_colors[self.node1] = 'b'
                self.plot_graph()  # 更新图形
                
            elif event.key() == Qt.Key_Down or event.key() == Qt.Key_S:
                self.node_colors[self.node1] = 'r' 
                if self.node2 is not None:
                    self.node_colors[self.node2] = 'b' 
                self.node1 = (self.node1 -1) % len(self.marks2)
                self.counter_Left_and_Right = None
                self.node_colors[self.node1] = 'b'
                self.plot_graph()  # 更新图形
            
            elif event.key() == Qt.Key_Left or event.key() == Qt.Key_A:
                if self.counter_Left_and_Right == None:
                    self.counter_Left_and_Right = 0  
                self.counter_Left_and_Right -=1
                if self.node2 is not None:
                    self.node_colors[self.node2] = 'r' 
                self.node_colors[self.node1] = 'b'
                nearest_indices = np.argsort(np.linalg.norm(self.marks2 - self.marks2[self.node1], axis=1))
                nearest_indices = nearest_indices[nearest_indices != self.node1][:5]
                # nearest_indices = nearest_indices[np.argsort(nearest_indices)]
                self.node2 = nearest_indices[self.counter_Left_and_Right %5]
                self.node_colors[self.node2] = 'b'
                self.plot_graph()  # 更新图形
            
            elif event.key() == Qt.Key_Right or event.key() == Qt.Key_D:
                if self.counter_Left_and_Right == None:
                    self.counter_Left_and_Right = -1   
                self.counter_Left_and_Right +=1
                if self.node2 is not None:
                    self.node_colors[self.node2] = 'r' 
                self.node_colors[self.node1] = 'b'
                nearest_indices = np.argsort(np.linalg.norm(self.marks2 - self.marks2[self.node1], axis=1))
                nearest_indices = nearest_indices[nearest_indices != self.node1][:5]
                # nearest_indices = nearest_indices[np.argsort(nearest_indices)]
                self.node2 = nearest_indices[self.counter_Left_and_Right %5]
                self.node_colors[self.node2] = 'b'
                self.plot_graph()  # 更新图形

            elif event.key() == Qt.Key_R or event.key() == Qt.Key_PageUp:
                if self.node2 is not None:
                    temp = self.node1
                    self.node1 = self.node2
                    self.node2 = temp
                    self.counter_Left_and_Right = None
                    self.plot_graph()  # 更新图形
                    
            elif event.key() == Qt.Key_F or event.key() == Qt.Key_PageDown:
                self.move_center_to_next_node()
            
            

















# 单独处理一个预备体
# 设置各种地址
import os

# 修改当前工作目录，以后输出文件只需要写文件名
new_dir = "D:/李娅宁/肩台外侧点-0715/"
os.chdir(new_dir)

# 设置输入文件路径
obj_file_path = 'Aug15/肩台外侧标志点_0815/0812-1-5_4.obj'
mark_file_path1 = 'Aug15/肩台外侧标志点_0815/0812-1-5_4.obj.mark'
mark_file_path2 = 'Aug12/多分类mark2/4.obj.mark'

obj_encoding = detect_encoding(obj_file_path)
mark_encoding1 = detect_encoding(mark_file_path1)
mark_encoding2 = detect_encoding(mark_file_path2)

obj_vertices, obj_faces = load_obj_file(obj_file_path, obj_encoding)
checked_vertices, faces = insert_midpoint_points(obj_vertices, obj_faces)
centered_vertices = center_vertices(checked_vertices)
marks1 = load_mark_file(mark_file_path1, mark_encoding1)
marks2 = load_mark_file(mark_file_path2, mark_encoding2)
marks = marks1 + marks2

aligned_points, final_rotation_matrix = align_point_cloud(np.asarray(centered_vertices))
aligned_marks2 = np.asarray(marks2).dot(final_rotation_matrix)

# 启动GUI
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 测试数据，正式使用时删掉这一行
    # aligned_marks2 = np.random.rand(10, 3) * 10
    
    # 真实数据，正式使用时要解除注释
    aligned_marks2 = np.asarray(marks2).dot(final_rotation_matrix)

    marks2_file_name = os.path.basename(mark_file_path2)
    
    editor = TopologyEditor(aligned_marks2, marks2_file_name)

    # 设置窗口标题为文件名
    editor.setWindowTitle(f"3D Graph - {marks2_file_name}")
    
    editor.show()
    
    sys.exit(app.exec_())

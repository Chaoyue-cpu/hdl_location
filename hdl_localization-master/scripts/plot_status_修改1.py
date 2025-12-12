#!/usr/bin/env python3
import rospy
import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt
from threading import Lock
from hdl_localization.msg import ScanMatchingStatus
import matplotlib
matplotlib.use('Qt5Agg')  # 必须在其他matplotlib导入前设置

class Plotter(object):
    def __init__(self):
        # 初始化图形界面
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)  # 为图例留出空间
        self.plot_lock = Lock()
        self.status_buffer = []
        
        # ROS订阅和定时器
        self.status_sub = rospy.Subscriber('/status', ScanMatchingStatus, self.status_callback)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        
        # 启用交互模式
        plt.ion()
        plt.show(block=False)

    def status_callback(self, status_msg):
        with self.plot_lock:
            self.status_buffer.append(status_msg)
            if len(self.status_buffer) > 100:
                self.status_buffer = self.status_buffer[-100:]

    def timer_callback(self, event):
        if len(self.status_buffer) < 2:
            return

        errors = {}
        for status in self.status_buffer:
            for label, error in zip(status.prediction_labels, status.prediction_errors):
                if label.data not in errors:
                    errors[label.data] = []

                quat = [error.rotation.x, error.rotation.y, 
                       error.rotation.z, error.rotation.w]
                trans = [error.translation.x, 
                        error.translation.y, 
                        error.translation.z]

                t = status.header.stamp.to_sec()
                t_error = np.linalg.norm(trans)
                r_error = np.linalg.norm(
                    scipy.spatial.transform.Rotation.from_quat(quat).as_rotvec()
                )

                if errors[label.data] and abs(errors[label.data][-1][0] - t) > 1.0:
                    errors[label.data] = []
                errors[label.data].append((t, t_error, r_error))

        self.update_plot(errors)

    def update_plot(self, errors):
        """线程安全的绘图更新"""
        with self.plot_lock:
            self.ax1.clear()
            self.ax2.clear()
            
            for label in errors:
                if not errors[label]:
                    continue
                    
                errs = np.array(errors[label])
                self.ax1.plot(errs[:, 0], errs[:, 1], label=label, linewidth=1.5)
                self.ax2.plot(errs[:, 0], errs[:, 2], label=label, linewidth=1.5)

            # 配置图表样式
            self.ax1.set_ylabel('Translation Error (m)', fontsize=10)
            self.ax2.set_ylabel('Rotation Error (rad)', fontsize=10)
            self.ax2.set_xlabel('Time (s)', fontsize=10)
            
            # 自动调整坐标轴范围
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            # 统一图例
            if errors:
                self.ax1.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.2),
                    ncol=min(4, len(errors)),
                    fontsize=9
                )
            
            plt.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()  # 替代start_event_loop

    def shutdown(self):
        """清理资源"""
        plt.close('all')

def main():
    rospy.init_node('status_plotter', anonymous=True)
    plotter = Plotter()
    
    try:
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        plotter.shutdown()

if __name__ == '__main__':
    main()
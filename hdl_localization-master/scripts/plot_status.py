#!/usr/bin/env python3
import rospy
import numpy as np
import scipy.spatial.transform
from matplotlib import pyplot as plt
from hdl_localization.msg import *
import threading
import time


class Plotter:
    def __init__(self):
        plt.ion()
        self.status_buffer = []
        self.errors = {}
        self.lock = threading.Lock()

        # ROS 初始化放到子线程中运行
        ros_thread = threading.Thread(target=self.ros_thread)
        ros_thread.start()

    def ros_thread(self):
        rospy.init_node('status_plotter')
        self.status_sub = rospy.Subscriber('/status', ScanMatchingStatus, self.status_callback)
        rospy.spin()

    # def status_callback(self, status_msg):
    #     with self.lock:
    #         self.status_buffer.append(status_msg)
    #         if len(self.status_buffer) > 50:
    #             self.status_buffer = self.status_buffer[-50:]
    def status_callback(self, status_msg):
        # 使用线程锁确保在多线程环境下对共享资源的安全访问
        with self.lock:
            # 将接收到的状态消息添加到状态缓冲区列表中
            self.status_buffer.append(status_msg)
            # 检查状态缓冲区的长度是否超过50
            if len(self.status_buffer) > 50:
                # 若超过50，则只保留列表中最后50个元素，实现缓冲区的滚动更新
                self.status_buffer = self.status_buffer[-50:]


    def process_and_plot(self):
        if len(self.status_buffer) < 2:
            return

        with self.lock:
            self.errors.clear()
            for status in self.status_buffer:
                for label, error in zip(status.prediction_labels, status.prediction_errors):
                    key = label.data
                    if key not in self.errors:
                        self.errors[key] = []

                    quat = [error.rotation.x, error.rotation.y, error.rotation.z, error.rotation.w]
                    trans = [error.translation.x, error.translation.y, error.translation.z]

                    t = status.header.stamp.secs + status.header.stamp.nsecs / 1e9
                    t_error = np.linalg.norm(trans)
                    r_error = np.linalg.norm(scipy.spatial.transform.Rotation.from_quat(quat).as_rotvec())

                    if self.errors[key] and abs(self.errors[key][-1][0] - t) > 1.0:
                        self.errors[key] = []

                    self.errors[key].append((t, t_error, r_error))

        # 绘图逻辑必须在主线程
        plt.clf()
        for label in self.errors:
            errs = np.array(self.errors[label])
            plt.subplot(211)
            plt.plot(errs[:, 0], errs[:, 1], label=label)

            plt.subplot(212)
            plt.plot(errs[:, 0], errs[:, 2], label=label)

        plt.subplot(211)
        plt.ylabel("trans error")
        plt.subplot(212)
        plt.ylabel("rot error")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(self.errors))
        plt.pause(0.1)


def main():
    rospy.init_node('status_plotter')  # 必须在主线程！
    plotter = Plotter()

    # ROS spin 放子线程没问题
    ros_thread = threading.Thread(target=rospy.spin)
    ros_thread.start()

    # 主线程做图形刷新
    while not rospy.is_shutdown():
        plotter.process_and_plot()
        time.sleep(0.1)



if __name__ == '__main__':
    main()

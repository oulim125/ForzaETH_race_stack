import os
import yaml
import subprocess
from pathlib import Path

import rclpy
from rclpy.node import Node

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Arrow

from ament_index_python.packages import get_package_share_directory
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import MarkerArray


def get_data_path(subpath: str = "") -> Path:
    """
    stack_master 패키지 기준 절대 경로로 변환.
    항상 (repo root)/race_stack/stack_master/<subpath> 를 가리키도록 고정.
    예) .../forza_ws/race_stack/stack_master/maps/<map>
    """
    sm_share = Path(get_package_share_directory("stack_master"))
    repo_root = sm_share.parents[3]           # .../forza_ws/race_stack
    return repo_root / "stack_master" / subpath


class SectorSlicer(Node):
    """
    /global_waypoints, /trackbounds/markers 를 받아 GUI로 섹터 구간을 자르고,
    다음 두 YAML을 생성한다.

      1) speed_scaling.yaml (기존 포맷, sector_tuner 네임스페이스)
      2) ctrl_sectors.yaml  (신규 포맷, 스위처 노드 네임스페이스)
         - 섹터마다 어떤 컨트롤러를 켤지(pp/tln) 지정

    * 섹터 단위는 '웨이포인트 인덱스' 기준으로 고정.
      (start/end 포함 범위, 첫 섹터 start=0, 이후 start=이전+1)
    """

    def __init__(self, future_done):
        super().__init__(
            "sector_slicer",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=False,
        )
        self.future_done = future_done

        # ---------- Parameters ----------
        self.declare_parameter("map_name", "hangar_1905_v0")
        map_name = self.get_parameter("map_name").get_parameter_value().string_value

        # 기본 저장 위치: <repo>/race_stack/stack_master/maps/<map_name>
        default_yaml_dir = str(get_data_path(f"maps/{map_name}"))
        self.declare_parameter("yaml_dir", default_yaml_dir)

        # ctrl_sectors.yaml 의 최상단 노드 이름(네임스페이스)
        self.declare_parameter("switcher_node_name", "sector_runtime_ctrl_switcher")

        # 섹터 기본 on/off 값
        self.declare_parameter("default_pp_enabled", True)
        self.declare_parameter("default_tln_enabled", False)

        # ---------- Members ----------
        self.yaml_dir = Path(self.get_parameter("yaml_dir").get_parameter_value().string_value)
        self.yaml_dir.mkdir(parents=True, exist_ok=True)

        self.switcher_node_name = self.get_parameter("switcher_node_name").get_parameter_value().string_value
        self.default_pp_enabled = bool(self.get_parameter("default_pp_enabled").get_parameter_value().bool_value)
        self.default_tln_enabled = bool(self.get_parameter("default_tln_enabled").get_parameter_value().bool_value)

        self.get_logger().info(f"YAML output dir: {self.yaml_dir}")
        self.get_logger().info(f"Switcher node (top key of ctrl_sectors.yaml): {self.switcher_node_name}")

        # 데이터 버퍼
        self.global_wpnts = None  # WpntArray
        self.track_bounds = None  # MarkerArray
        self.sector_pnts = [0]    # 섹터 분할 인덱스 목록; 항상 0부터 시작
        self.glob_slider_idx = 0  # GUI 슬라이더 현재 인덱스
        self.speed_scaling = 0.5  # 기존 포맷 호환 (필요 시 조절 가능)

        # ---------- Subscriptions ----------
        self.create_subscription(WpntArray, "/global_waypoints", self.global_wpnts_cb, 10)
        self.create_subscription(MarkerArray, "/trackbounds/markers", self.bounds_cb, 10)

        # ---------- Timer ----------
        # 데이터가 준비되면 GUI 한 번만 띄우고, 완료 시 YAML 저장 & 후처리
        self.timer = self.create_timer(0.5, self._maybe_start_gui_once)
        self._gui_started = False

    # -------------------- ROS Callbacks --------------------

    def global_wpnts_cb(self, msg: WpntArray):
        self.global_wpnts = msg

    def bounds_cb(self, msg: MarkerArray):
        self.track_bounds = msg

    def _maybe_start_gui_once(self):
        if self._gui_started:
            return
        if self.global_wpnts is None or self.track_bounds is None:
            return

        self._gui_started = True
        self.timer.cancel()  # GUI는 한 번만

        self.get_logger().info("Starting sector GUI…")
        self._run_sector_gui()

        # GUI 종료 후 섹터 인덱스 확정
        self.get_logger().info(f"Selected Sector IDXs: {self.sector_pnts}")

        # YAML 생성
        self._dump_speed_scaling_yaml()
        self._dump_ctrl_sectors_yaml()

        # 후처리(빌드/설치 등)
        self._invoke_finish_script()

        # 작업 종료 신호
        self.future_done.set_result(True)

    # -------------------- GUI --------------------

    def _run_sector_gui(self):
        xs = np.array([wp.x_m for wp in self.global_wpnts.wpnts])
        ys = np.array([wp.y_m for wp in self.global_wpnts.wpnts])
        n  = len(xs)

        fig, (ax_plot, axslider, ax_select, ax_finish) = plt.subplots(
            4, 1, gridspec_kw={"height_ratios": [5, 1, 1, 1]}
        )

        def draw_base():
            ax_plot.cla()
            ax_plot.plot(xs, ys, "r-", linewidth=0.7)
            bx = [m.pose.position.x for m in self.track_bounds.markers]
            by = [m.pose.position.y for m in self.track_bounds.markers]
            ax_plot.plot(bx, by, "g-", linewidth=0.4)
            ax_plot.grid()
            ax_plot.set_aspect("equal", "datalim")
            ax_plot.set_xlabel("east [m]")
            ax_plot.set_ylabel("north [m]")
            if n >= 2:
                arr = Arrow(xs[0], ys[0], 10*(xs[1]-xs[0]), 10*(ys[1]-ys[0]), color="gray", width=0.5)
                ax_plot.add_artist(arr)

        def draw_current(idx: int):
            draw_base()
            idx = max(0, min(n-1, idx))
            ax_plot.scatter(xs[idx], ys[idx], s=30)
            sel = [p for p in self.sector_pnts if 0 <= p < n]
            if sel:
                ax_plot.scatter(xs[sel], ys[sel], c="red", s=20)

        def on_slide(val):
            idx = int(slider.val)
            self.glob_slider_idx = idx
            draw_current(idx)
            fig.canvas.draw_idle()

        def on_select(event):
            self.sector_pnts.append(self.glob_slider_idx)
            self.sector_pnts = sorted(list(set(self.sector_pnts)))
            draw_current(self.glob_slider_idx)
            fig.canvas.draw_idle()

        def on_finish(event):
            self.sector_pnts.append(n)
            self.sector_pnts = sorted(list(set(self.sector_pnts)))
            plt.close()

        draw_current(0)
        slider = Slider(axslider, "Index", 0, n-1, valinit=0, valfmt="%d")
        slider.on_changed(on_slide)

        btn_select = Button(ax_select, "Select")
        btn_select.on_clicked(on_select)

        btn_finish = Button(ax_finish, "Done")
        btn_finish.on_clicked(on_finish)

        plt.show()

    # -------------------- YAML Dumpers --------------------

    def _dump_speed_scaling_yaml(self):
        points    = self.sector_pnts
        n_sectors = max(0, len(points) - 1)

        params = {"global_limit": self.speed_scaling, "n_sectors": n_sectors}
        for i in range(n_sectors):
            start = points[i] if i == 0 else points[i] + 1
            end   = points[i + 1]
            params[f"Sector{i}"] = {
                "start": int(start),
                "end":   int(end),
                "scaling": float(self.speed_scaling),
                "only_FTG": False,
                "no_FTG":   False,
            }

        data = {"sector_tuner": {"ros__parameters": params}}
        out_path = self.yaml_dir / "speed_scaling.yaml"
        with open(out_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        self.get_logger().info(f"Dumped: {out_path}")

    def _dump_ctrl_sectors_yaml(self):
        points    = self.sector_pnts
        n_sectors = max(0, len(points) - 1)

        params = {"n_sectors": n_sectors}
        for i in range(n_sectors):
            start = points[i] if i == 0 else points[i] + 1
            end   = points[i + 1]
            params[f"Sector{i}"] = {
                "start": int(start),
                "end":   int(end),
                "controllers": {
                    "pp":  bool(self.default_pp_enabled),
                    "tln": bool(self.default_tln_enabled),
                },
            }

        data = {self.switcher_node_name: {"ros__parameters": params}}
        out_path = self.yaml_dir / "ctrl_sectors.yaml"
        with open(out_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        self.get_logger().info(f"Dumped: {out_path} (top={self.switcher_node_name})")

    # -------------------- Post Step --------------------

    def _invoke_finish_script(self):
        self.get_logger().info("Invoking finish script (ros2 run sector_tuner finish_sector.sh)")
        subprocess.Popen("ros2 run sector_tuner finish_sector.sh", shell=True)
        self.get_logger().info("Done Slicing")


def main():
    rclpy.init()
    fut = rclpy.Future()
    node = SectorSlicer(fut)
    rclpy.spin_until_future_complete(node, fut)
    rclpy.shutdown()


if __name__ == "__main__":
    main()


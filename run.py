"""一键运行脚本：支持用户注册/登录/权限管理 + 调用 main.py。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from auth import ADMIN_PASSWORD, ADMIN_USERNAME, AuthService
from config import (
    CLASSES_SPEC,
    COUNT_LINE,
    DEFAULT_DB_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    EDGE_IGNORE_PX,
    IMGSZ,
    LINE1,
    LINE2,
    LINE_DISTANCE_M,
    LLM_PAYLOAD_NAME,
    LOST_HOLD,
    MOTO_CONF,
    PERSON_CONF,
    PERSON_MIN_FRAMES,
    SPEED_DIRECTION_MODE,
    STABLE_KEEP_RATIO,
    STRIDE,
    CROSS_TOLERANCE,
    TRAIL_LEN,
    TRACKER_CFG,
    VEH_CONF,
    VEHICLE_DEDUPE_IOU,
    VEHICLE_MIN_AREA_RATIO,
    VEHICLE_MIN_FRAMES,
    YOLO_CONF,
    YOLO_IOU,
    YOLO_MODEL,
)


def must_exist(file_path: Path, desc: str) -> None:
    """检查文件是否存在，不存在则退出。"""
    if not file_path.exists():
        print(f"[错误] {desc}不存在：{file_path}")
        raise SystemExit(1)


def is_ultralytics_builtin_model(model_path: str) -> bool:
    """判断是否为 Ultralytics 可自动下载的官方模型名。"""
    name = Path(model_path).name.lower()
    return name.endswith(".pt") and name.startswith("yolo11")


def parse_xyxy_arg(value: str):
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("坐标格式应为 x1,y1,x2,y2")
    try:
        return tuple(int(float(p)) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("坐标必须为数字") from exc


def build_main_command(args, project_dir: Path) -> list[str]:
    model_file = project_dir / args.model_path if not Path(args.model_path).is_absolute() else Path(args.model_path)
    input_video = (
        (project_dir / args.input_video).resolve()
        if not Path(args.input_video).is_absolute()
        else Path(args.input_video)
    )

    output_dir = project_dir / DEFAULT_OUTPUT_DIR
    report_dir = project_dir / DEFAULT_REPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / "output.mp4"
    counts_csv = output_dir / "counts.csv"
    events_csv = output_dir / "events.csv"
    traffic_csv = output_dir / "traffic.csv"
    anomalies_csv = output_dir / "anomalies.csv"
    llm_payload_json = output_dir / LLM_PAYLOAD_NAME

    print("[信息] 开始检查运行条件...")
    model_arg = str(model_file)
    if model_file.exists():
        pass
    elif is_ultralytics_builtin_model(args.model_path):
        print(f"[信息] 检测到官方模型名 {args.model_path}，若本地不存在将由 Ultralytics 自动下载。")
        model_arg = str(args.model_path)
    else:
        must_exist(model_file, "模型文件")
    must_exist(input_video, "输入视频")

    cmd = [
        sys.executable,
        "main.py",
        "--input-vid",
        str(input_video),
        "--save-path",
        str(output_video),
        "--counts-csv",
        str(counts_csv),
        "--events-csv",
        str(events_csv),
        "--traffic-csv",
        str(traffic_csv),
        "--anomalies-csv",
        str(anomalies_csv),
        "--llm-payload-json",
        str(llm_payload_json),
        "--reports-dir",
        str(report_dir),
        "--stride",
        str(max(1, int(args.stride))),
        "--model-path",
        model_arg,
        "--tracker",
        str(args.tracker),
        "--yolo-conf",
        str(float(args.yolo_conf)),
        "--yolo-iou",
        str(float(args.yolo_iou)),
        "--imgsz",
        str(int(args.imgsz)),
        "--classes",
        str(args.classes),
        "--veh-conf",
        str(float(args.veh_conf)),
        "--moto-conf",
        str(float(args.moto_conf)),
        "--person-conf",
        str(float(args.person_conf)),
        "--person-min-frames",
        str(int(args.person_min_frames)),
        "--vehicle-min-frames",
        str(int(args.vehicle_min_frames)),
        "--vehicle-min-area-ratio",
        str(float(args.vehicle_min_area_ratio)),
        "--edge-ignore-px",
        str(int(args.edge_ignore_px)),
        "--vehicle-dedupe-iou",
        str(float(args.vehicle_dedupe_iou)),
        "--lost-hold",
        str(int(args.lost_hold)),
        "--stable-keep-ratio",
        str(float(args.stable_keep_ratio)),
        "--trail-len",
        str(int(args.trail_len)),
        "--trail-mode",
        str(args.trail_mode),
        "--highlight-ids",
        str(args.highlight_ids),
        "--count-line",
        ",".join(str(x) for x in args.count_line),
        "--speed-line-1",
        ",".join(str(x) for x in args.speed_line_1),
        "--speed-line-2",
        ",".join(str(x) for x in args.speed_line_2),
        "--line-distance-m",
        str(float(args.line_distance_m)),
        "--speed-direction-mode",
        str(args.speed_direction_mode),
        "--cross-tolerance",
        str(int(args.cross_tolerance)),
    ]
    if args.no_show:
        cmd.append("--no-show")

    return cmd


def run_monitor_task(args, project_dir: Path) -> None:
    cmd = build_main_command(args, project_dir)
    print("[信息] 检查通过，开始执行 YOLO11 + ByteTrack 检测与跟踪。")
    subprocess.run(cmd, cwd=str(project_dir), check=True)

    output_dir = project_dir / DEFAULT_OUTPUT_DIR
    report_dir = project_dir / DEFAULT_REPORT_DIR
    print("\n[完成] 运行结束，输出文件如下：")
    print(f"- 视频文件: {output_dir / 'output.mp4'}")
    print(f"- 车流统计: {output_dir / 'counts.csv'}")
    print(f"- 过线事件: {output_dir / 'events.csv'}")
    print(f"- 车速明细: {output_dir / 'traffic.csv'}")
    print(f"- 异常提示: {output_dir / 'anomalies.csv'}")
    print(f"- 大模型载荷: {output_dir / LLM_PAYLOAD_NAME}")
    print(f"- 综合图表: {report_dir / '综合统计图.png'}")


def show_results(project_dir: Path) -> None:
    output_dir = project_dir / DEFAULT_OUTPUT_DIR
    report_dir = project_dir / DEFAULT_REPORT_DIR
    print("\n[结果文件]")
    for p in [
        output_dir / "output.mp4",
        output_dir / "counts.csv",
        output_dir / "events.csv",
        output_dir / "traffic.csv",
        report_dir / "综合统计图.png",
    ]:
        status = "存在" if p.exists() else "未生成"
        print(f"- {p} [{status}]")


def prompt_input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def prompt_password(prompt: str) -> str:
    # 为简化演示，命令行直接输入；如需隐藏可替换为 getpass.getpass。
    return prompt_input(prompt)


def handle_register(auth: AuthService) -> None:
    print("\n=== 用户注册 ===")
    username = prompt_input("用户名: ")
    pwd = prompt_password("密码: ")
    pwd2 = prompt_password("确认密码: ")
    ok, msg = auth.register(username, pwd, pwd2)
    print(f"[{'成功' if ok else '失败'}] {msg}")


def handle_login(auth: AuthService):
    print("\n=== 用户登录 ===")
    username = prompt_input("用户名: ")
    pwd = prompt_password("密码: ")
    ok, msg, user = auth.login(username, pwd)
    print(f"[{'成功' if ok else '失败'}] {msg}")
    return user if ok else None


def show_users(auth: AuthService) -> None:
    print("\n=== 用户列表 ===")
    users = auth.list_users()
    if not users:
        print("当前没有用户。")
        return
    print("ID\t用户名\t角色\t创建时间")
    for u in users:
        print(f"{u.id}\t{u.username}\t{u.role}\t{u.created_at}")


def delete_user_flow(auth: AuthService, current_user) -> None:
    target = prompt_input("请输入要删除的普通用户名: ")
    ok, msg = auth.delete_user(current_user.role, current_user.username, target)
    print(f"[{'成功' if ok else '失败'}] {msg}")


def user_session_loop(auth: AuthService, current_user, args, project_dir: Path) -> None:
    while True:
        print(f"\n=== 已登录：{current_user.username} ({current_user.role}) ===")
        print("1. 运行监控任务")
        print("2. 查看统计结果")
        if current_user.role == "admin":
            print("3. 查看全部用户")
            print("4. 删除普通用户")
            print("5. 退出登录")
        else:
            print("3. 退出登录")

        choice = prompt_input("请选择: ")
        if choice == "1":
            run_monitor_task(args, project_dir)
        elif choice == "2":
            show_results(project_dir)
        elif current_user.role == "admin" and choice == "3":
            show_users(auth)
        elif current_user.role == "admin" and choice == "4":
            delete_user_flow(auth, current_user)
        elif (current_user.role == "admin" and choice == "5") or (current_user.role != "admin" and choice == "3"):
            print("[信息] 已退出登录。")
            return
        else:
            print("[提示] 无效选项，请重试。")


def auth_menu_loop(auth: AuthService, args, project_dir: Path) -> None:
    while True:
        print("\n=== 智能监控系统 ===")
        print("1. 注册")
        print("2. 登录")
        print("3. 退出")
        choice = prompt_input("请选择: ")

        if choice == "1":
            handle_register(auth)
        elif choice == "2":
            user = handle_login(auth)
            if user is not None:
                user_session_loop(auth, user, args, project_dir)
        elif choice == "3":
            print("[信息] 已退出程序。")
            return
        else:
            print("[提示] 无效选项，请重试。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行 YOLO11 + ByteTrack（含用户认证）")
    parser.add_argument("input_video", nargs="?", default="./video.mp4", help="输入视频路径（位置参数）")
    parser.add_argument("--stride", type=int, default=STRIDE, help="跳帧参数")
    parser.add_argument("--no-show", action="store_true", help="关闭窗口显示")
    parser.add_argument("--model-path", type=str, default=YOLO_MODEL, help="YOLO11 模型路径")
    parser.add_argument("--tracker", type=str, default=TRACKER_CFG, help="ByteTrack 配置路径")
    parser.add_argument(
        "--classes",
        type=str,
        default=CLASSES_SPEC,
        help='类别配置：person_vehicle | vehicle | person | all 或自定义 "0,1,2"',
    )
    parser.add_argument("--yolo-conf", type=float, default=YOLO_CONF, help="YOLO11 一阶段置信度阈值")
    parser.add_argument("--yolo-iou", type=float, default=YOLO_IOU, help="YOLO11 IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="YOLO11 推理分辨率")
    parser.add_argument("--veh-conf", type=float, default=VEH_CONF, help="车辆类别阈值")
    parser.add_argument("--moto-conf", type=float, default=MOTO_CONF, help="摩托车类别阈值")
    parser.add_argument("--person-conf", type=float, default=PERSON_CONF, help="行人类别阈值")
    parser.add_argument("--person-min-frames", type=int, default=PERSON_MIN_FRAMES, help="行人轨迹最小连续帧数")
    parser.add_argument("--vehicle-min-frames", type=int, default=VEHICLE_MIN_FRAMES, help="车辆轨迹最小连续帧数")
    parser.add_argument("--vehicle-min-area-ratio", type=float, default=VEHICLE_MIN_AREA_RATIO, help="车辆最小面积比例")
    parser.add_argument("--edge-ignore-px", type=int, default=EDGE_IGNORE_PX, help="边缘抑制像素阈值")
    parser.add_argument("--vehicle-dedupe-iou", type=float, default=VEHICLE_DEDUPE_IOU, help="车辆同帧去重IoU阈值")
    parser.add_argument("--lost-hold", type=int, default=LOST_HOLD, help="短时丢失保留帧数")
    parser.add_argument("--stable-keep-ratio", type=float, default=STABLE_KEEP_RATIO, help="已建立轨迹的低阈值保活比例")
    parser.add_argument("--trail-len", type=int, default=TRAIL_LEN, help="轨迹显示长度")
    parser.add_argument("--trail-mode", choices=["none", "all", "selected"], default="all", help="轨迹显示模式")
    parser.add_argument("--highlight-ids", type=str, default="", help='高亮 track_id，例如 "12" 或 "12,15"')
    parser.add_argument("--count-line", type=parse_xyxy_arg, default=COUNT_LINE, help='统计线坐标，例如 "100,360,900,360"')
    parser.add_argument("--speed-line-1", type=parse_xyxy_arg, default=LINE1, help='测速线1坐标，例如 "100,300,900,300"')
    parser.add_argument("--speed-line-2", type=parse_xyxy_arg, default=LINE2, help='测速线2坐标，例如 "100,420,900,420"')
    parser.add_argument("--line-distance-m", type=float, default=LINE_DISTANCE_M, help="两条测速线实际距离（米）")
    parser.add_argument(
        "--speed-direction-mode",
        choices=["vertical", "horizontal"],
        default=SPEED_DIRECTION_MODE,
        help="测速方向模式",
    )
    parser.add_argument("--cross-tolerance", type=int, default=CROSS_TOLERANCE, help="穿线容差像素")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="用户数据库 SQLite 路径")
    parser.add_argument("--no-auth", action="store_true", help="禁用登录认证，直接运行监控任务")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent

    if args.no_auth or (not sys.stdin.isatty()):
        if not args.no_auth and (not sys.stdin.isatty()):
            print("[提示] 检测到非交互终端，自动跳过登录流程（等效 --no-auth）。")
        run_monitor_task(args, project_dir)
        return

    auth = AuthService(db_path=str((project_dir / args.db_path).resolve()))
    print(f"[信息] 用户数据库: {(project_dir / args.db_path).resolve()}")
    print(f"[信息] 默认管理员账号: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    auth_menu_loop(auth, args, project_dir)


if __name__ == "__main__":
    main()
